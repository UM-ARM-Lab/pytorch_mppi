import copy

import torch

from arm_pytorch_utilities import linalg, handle_batch_input, sort_nicely, cache
import matplotlib.colors
from matplotlib import pyplot as plt
import os

from pytorch_mppi import mppi
import pytorch_seed
import logging
import collections

def is_sequence(obj):
    if isinstance(obj, str):
        return False
    return isinstance(obj, collections.abc.Sequence)


plt.switch_backend('Qt5Agg')

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


class LinearDeltaDynamics:
    def __init__(self, B):
        self.B = B

    @handle_batch_input(n=2)
    def __call__(self, state, action):
        nx = state + action @ self.B.transpose(0, 1)
        return nx


class ScaledLinearDynamics:
    def __init__(self, cost, B):
        self.B = B
        self.cost = cost

    @handle_batch_input(n=2)
    def __call__(self, state, action):
        nx = state + action @ self.B.transpose(0, 1) / torch.log(self.cost(state) + 1e-8).reshape(-1, 1) * 2
        return nx


class LQRCost:
    def __init__(self, Q, R, goal):
        self.Q = Q
        self.R = R
        self.goal = goal

    @handle_batch_input(n=2)
    def __call__(self, state, action=None):
        dx = self.goal - state
        c = linalg.batch_quadratic_product(dx, self.Q)
        if action is not None:
            c += linalg.batch_quadratic_product(action, self.R)
        return c


class HillCost:
    def __init__(self, Q, center, cost_at_center=1):
        self.Q = Q
        self.center = center
        self.cost_at_center = cost_at_center

    @handle_batch_input(n=2)
    def __call__(self, state, action=None):
        dx = self.center - state
        d = linalg.batch_quadratic_product(dx, self.Q)
        c = self.cost_at_center * torch.exp(-d)
        return c


class Toy2DEnvironment:
    def __init__(self, start=None, goal=None, dtype=torch.double, device="cpu", evaluate_running_cost=True,
                 visualize=True,
                 num_trajectories=5,
                 terminal_scale=100,
                 r=0.01):
        self.d = device
        self.dtype = dtype
        self.state_ranges = [
            (-5, 5),
            (-5, 5)
        ]
        self.evaluate_running_cost = evaluate_running_cost
        self.num_trajectories = num_trajectories
        self.visualize = visualize
        self.nx = 2

        self.start = start or torch.tensor([-3, -2], device=self.d, dtype=self.dtype)
        self.goal = goal or torch.tensor([2, 2], device=self.d, dtype=self.dtype)
        self.state = self.start

        self.costs = []

        eye = torch.eye(2, device=self.d, dtype=self.dtype)
        goal_cost = LQRCost(eye, eye * r, self.goal)
        self.costs.append(goal_cost)

        # for increasing difficulty, we add some "hills"
        self.costs.append(HillCost(torch.tensor([[0.1, 0.05], [0.05, 0.1]], device=self.d, dtype=self.dtype) * 2.5,
                                   torch.tensor([-0.5, -1.], device=self.d, dtype=self.dtype), cost_at_center=200))

        B = torch.tensor([[0.5, 0], [0, -0.5]], device=self.d, dtype=self.dtype)
        self.dynamics = LinearDeltaDynamics(B)
        # self.dynamics = ScaledLinearDynamics(self.running_cost, B)

        self.terminal_scale = terminal_scale

        self.trajectory_artist = None

        self.start_visualization()

    def reset(self):
        self.state = self.start
        self.clear_artist(self.rollout_artist)
        self.rollout_artist = None
        self.clear_trajectory()
        self.trajectory_artist = None

    def step(self, action):
        self.state = self.dynamics(self.state, action)
        return self.state

    def terminal_cost(self, states, actions):
        return self.terminal_scale * self.running_cost(states[..., -1, :])

    @handle_batch_input(n=2)
    def running_cost(self, state, action=None):
        c = None
        for cost in self.costs:
            if c is None:
                c = cost(state, action)
            else:
                c += cost(state, action)
        return c

    def start_visualization(self):
        if self.visualize:
            plt.ion()
            plt.show()

            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.ax.set_aspect('equal')
            self.ax.set(xlim=self.state_ranges[0])
            self.ax.set(ylim=self.state_ranges[0])

            self.cmap = "Greys"
            # artists for clearing / redrawing
            self.start_artist = None
            self.goal_artist = None
            self.cost_artist = None
            self.rollout_artist = None
            self.draw_costs()
            # self.draw_start()
            self.draw_goal()

    @staticmethod
    def get_v(i, values, rollouts):
        if type(values) != str and is_sequence(values) and len(values) == len(rollouts):
            c = values[i]
        else:
            c = values
        return c

    def draw_rollouts(self, rollouts, color="skyblue", label=None, end_state_color="tab:red", linewidth=1.5):
        if not self.visualize:
            return
        self.clear_artist(self.rollout_artist)
        artists = []
        for i, rollout in enumerate(rollouts):
            # prepend start state
            rollout = torch.cat([self.state.view(1, -1), rollout], dim=0)
            r = rollout.cpu()
            artists += [self.ax.scatter(r[0, 0], r[0, 1], color="tab:blue")]
            # if color is a string treat it as a single color, otherwise treat it as a list of colors
            artists += self.ax.plot(r[:, 0], r[:, 1],
                                    color=self.get_v(i, color, rollouts),
                                    label=self.get_v(i, label, rollouts),
                                    linewidth=self.get_v(i, linewidth, rollouts))
            artists += [self.ax.scatter(r[-1, 0], r[-1, 1], color=self.get_v(i, end_state_color, rollouts))]
        self.rollout_artist = artists
        if label is not None:
            self.ax.legend(loc = "upper right")
        plt.pause(0.001)

    def draw_trajectory_step(self, prev_state, cur_state, color="tab:blue"):
        if not self.visualize:
            return
        if self.trajectory_artist is None:
            self.trajectory_artist = []
        artists = self.trajectory_artist
        artists += self.ax.plot([prev_state[0].cpu(), cur_state[0].cpu()],
                                [prev_state[1].cpu(), cur_state[1].cpu()], color=color)
        plt.draw()
        plt.pause(0.001)

    def clear_trajectory(self):
        self.clear_artist(self.trajectory_artist)

    def draw_costs(self, resolution=0.05, value_padding=0):
        if not self.visualize:
            return
        coords = [torch.arange(low, high + resolution, resolution, dtype=self.dtype, device=self.d) for low, high in
                  self.state_ranges]
        pts = torch.cartesian_prod(*coords)
        val = self.running_cost(pts)

        norm = matplotlib.colors.Normalize(vmin=val.min().cpu() - value_padding, vmax=val.max().cpu())

        x = coords[0].cpu()
        z = coords[1].cpu()
        v = val.reshape(len(x), len(z)).transpose(0, 1).cpu()

        self.clear_artist(self.cost_artist)
        a = []
        a.append(self.ax.contourf(x, z, v, levels=[2, 4, 8, 16, 24, 32, 40, 50, 60, 80, 100, 150, 200, 250], norm=norm,
                                  cmap=self.cmap))
        # a.append(self.ax.contourf(x, z, v, norm=norm,
        #                           cmap=self.cmap))
        # reduce opacity
        a.append(self.ax.contour(x, z, v, levels=a[0].levels, colors='k', linestyles='dashed'))
        a.append(self.ax.clabel(a[1], a[1].levels, inline=True, fontsize=13))
        a[1].set_alpha(0.3)
        self.cost_artist = a

        plt.draw()
        plt.pause(0.0005)

    @staticmethod
    def clear_artist(artist):
        if artist is not None:
            for a in artist:
                a.remove()

    def draw_start(self):
        if not self.visualize:
            return
        self.clear_artist(self.start_artist)
        self.start_artist = self.draw_state(self.start, "tab:blue", label='start')

    def draw_goal(self):
        # when combined with other costs it's no longer the single goal so no need for label
        return
        if not self.visualize:
            return
        self.clear_artist(self.goal_artist)
        # when combined with other costs it's no longer the single goal so no need for label
        self.goal_artist = self.draw_state(self.goal, "tab:green")  # , label='goal')

    def draw_state(self, state, color, label=None, ox=-0.3, oy=0.3):
        artists = [self.ax.scatter(state[0].cpu(), state[1].cpu(), color=color)]
        if label is not None:
            artists.append(self.ax.text(state[0].cpu() + ox, state[1].cpu() + oy, label, color=color))
        plt.pause(0.0001)
        return artists


def make_gif(imgs_dir, gif_name):
    import imageio
    images = []
    # human sort
    names = os.listdir(imgs_dir)
    sort_nicely(names)
    for filename in names:
        if filename.endswith(".png"):
            images.append(imageio.v2.imread(os.path.join(imgs_dir, filename)))
    imageio.mimsave(gif_name, images, duration=0.1)


def make_gif_ffmpeg(imgs_dir, gif_name, fps=6):
    import subprocess
    # first generate palette and then use it to generate gif
    palette_path = os.path.join("images", "palette.png")
    cmd = ["ffmpeg", "-y", "-i", os.path.join(imgs_dir, "%d.png"), "-vf", "palettegen", palette_path]
    subprocess.run(cmd)
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(imgs_dir, "%d.png"), "-i", palette_path,
           "-lavfi", "paletteuse", gif_name]
    subprocess.run(cmd)


def do_control(env, ctrl, ch, seeds=(0,), run_steps=20, num_refinement_steps=1, save_img=True, plot_single=False,
               evaluate_running_cost=True, plot_trajectory_candidates=False):
    if save_img:
        os.makedirs("images", exist_ok=True)
        os.makedirs("images/runs", exist_ok=True)
        os.makedirs("images/gif", exist_ok=True)

    for seed in seeds:
        pytorch_seed.seed(seed)
        key = f"{ctrl.__class__.__name__}"

        # use the same nominal trajectory to start with for all the evaluations for fairness
        # parameters for our sample evaluation function - lots of choices for the evaluation function
        rollout_costs = []
        actual_costs = []
        controls = []
        state = env.state
        # we sample multiple trajectories for the same start to goal problem, but in your case you should consider
        # evaluating over a diverse dataset of trajectories
        for i in range(run_steps):
            # mppi.U = nominal_trajectory.clone()
            # the nominal trajectory at the start will be different if the horizon's changed
            # mppi.change_horizon(mppi.T)
            # usually MPPI will have its nominal trajectory warm-started from the previous iteration
            # for a fair test of tuning we will reset its nominal trajectory to the same random one each time
            # we manually warm it by refining it for some steps
            u = None
            for k in range(num_refinement_steps):
                last_refinement = k == num_refinement_steps - 1
                u = ctrl.command(state, shift_nominal_trajectory=last_refinement)

            rollout = ctrl.get_rollouts(state)

            rollout_cost = 0
            this_cost = env.running_cost(state)
            rollout = rollout[0]
            # here we evaluate on the rollout MPPI cost of the resulting trajectories
            # alternative costs for tuning the parameters are possible, such as just considering terminal cost
            if evaluate_running_cost:
                for t in range(len(rollout) - 1):
                    rollout_cost = rollout_cost + env.running_cost(rollout[t], ctrl.U[t])
            rollout_cost = rollout_cost + env.terminal_cost(rollout, ctrl.U)

            prev_state = copy.deepcopy(state)
            state = env.step(u)

            if plot_trajectory_candidates:
                from matplotlib import cm
                # only plot some candidates rather than all of them
                num_candidates = min(10, ctrl.K)
                # for the combined trajectory
                color = []
                end_color = []
                rollouts = []
                linewidth = []
                labels = []
                # for all the candidates
                # create matplotlib color map based on cost
                cost = ctrl.cost_total.cpu()
                best_idx = torch.argsort(cost)

                norm = matplotlib.colors.Normalize(vmin=cost.min(), vmax=cost[best_idx][:num_candidates*5].max())
                m = cm.ScalarMappable(norm=norm, cmap=cm.jet)

                traj_color = m.to_rgba(cost)
                # lower alpha
                traj_color[:, 3] = 0.2

                # get rollouts per sampled action trajectory
                for j in range(num_candidates):
                    idx = best_idx[j]
                    this_U = ctrl.actions[0, idx]
                    rollouts.append(ctrl.get_rollouts(state, U=this_U)[0])
                    color.append(traj_color[idx])
                    end_color.append([1, 0, 0, 0.2])
                    linewidth.append(1)
                    labels.append(None)
                color.append("skyblue")
                end_color.append("tab:red")
                rollouts.append(rollout)
                linewidth.append(2)
                labels.append(key)
                env.draw_rollouts(rollouts, color=color, end_state_color=end_color, linewidth=linewidth, label=labels)
            else:
                # just draw the single state rollout
                env.draw_rollouts([rollout], label=key)

            env.draw_trajectory_step(prev_state, state)

            if save_img:
                plt.savefig(f"images/runs/{i}.png")

            print(f"step {i} state {state} current cost {this_cost} rollout cost {rollout_cost}")
            actual_costs.append(this_cost.cpu())
            rollout_costs.append(rollout_cost.cpu())
            controls.append(u.cpu())

        controls = torch.stack(controls)
        # consider total difference
        control_diff = torch.diff(controls, dim=0)
        print(f"total accumulated cost: {sum(actual_costs)}")
        print(f"total accumulated rollout cost: {sum(rollout_costs)}")
        env.reset()
        ctrl.reset()

        secondary_key = (seed, ctrl.get_params())
        # make_gif("images/runs", f"images/gif/{key}_{seed}.gif")
        make_gif_ffmpeg("images/runs", f"images/gif/{key}_{seed}.gif", fps=10)
        if key not in ch:
            ch[key] = {}
        ch[key][secondary_key] = {
            "actual_costs": actual_costs,
            "rollout_costs": rollout_costs,
            "controls": controls,
            "control_diff": control_diff,
        }
        ch.save()

        if plot_single:
            # plot the costs with the step
            fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(8, 14))
            # xlim 0
            ax[0].set_xlim(0, run_steps - 1)
            # tick on x for every step discretely
            ax[0].plot(actual_costs)
            ax[0].set_title(f"actual costs total: {sum(actual_costs)}")
            ax[0].set_ylim(0, max(actual_costs) * 1.1)
            ax[1].plot(rollout_costs)
            ax[1].set_title(f"rollout costs total: {sum(rollout_costs)}")
            # set the y axis to be log scale
            ax[1].set_yscale('log')

            ax[2].plot(controls[:, 0], label='u0')
            ax[2].plot(controls[:, 1], label='u1')
            ax[2].legend()
            ax[2].set_title(f"control inputs total diff: {control_diff.abs().sum()}")
            ax[2].set_xticks(range(run_steps))
            plt.tight_layout()
            plt.pause(0.001)
            input("Press Enter to close the window and exit...")


def plot_result(ch):
    num_steps = 20

    def simplify_name(name):
        # remove shared parameters
        return name.replace("K=500 T=20 M=1 ", "").replace("noise_mu=[0. 0.] noise_sigma=[[1. 0.], [0. 1.]]",
                                                           "").replace("_lambda=1 ", " ")

    methods = {}
    for key, values in ch.items():
        for secondary_key, data in values.items():
            actual_costs = torch.tensor(data["actual_costs"])
            rollout_costs = torch.tensor(data["rollout_costs"])
            controls = data["controls"]
            control_diff = data["control_diff"]

            method_name = f"{key}_{secondary_key[1]}"
            if method_name not in methods:
                methods[method_name] = {
                    "actual_costs": [actual_costs],
                    "rollout_costs": [rollout_costs],
                    "controls": [controls],
                    "control_diff": [control_diff],
                }
            else:
                m = methods[method_name]
                m["actual_costs"].append(actual_costs)
                m["rollout_costs"].append(rollout_costs)
                m["controls"].append(controls)
                m["control_diff"].append(control_diff)

    method_names = "\n".join(methods.keys())
    print(f"all method keys\n{method_names}")

    allowed_names = [
        # "MPPI_K=500 T=20 M=1 lambda=1 noise_mu=[0. 0.] noise_sigma=[[1. 0.], [0. 1.]]",
        # "SMPPI_K=500 T=20 M=1 lambda=1 noise_mu=[0. 0.] noise_sigma=[[1. 0.], [0. 1.]] w=5 t=1.0",
        # "SMPPI_K=500 T=20 M=1 lambda=10 noise_mu=[0. 0.] noise_sigma=[[1. 0.], [0. 1.]] w=10 t=1.0",
        # "KMPPI_K=500 T=20 M=1 lambda=1 noise_mu=[0. 0.] noise_sigma=[[1. 0.], [0. 1.]] num_support_pts=5 kernel=rbf4theta",
        # "KMPPI_K=500 T=20 M=1 lambda=1 noise_mu=[0. 0.] noise_sigma=[[1. 0.], [0. 1.]] num_support_pts=5 kernel=RBFKernel(sigma=1.5)",
        "KMPPI_K=500 T=20 M=1 lambda=1 noise_mu=[0. 0.] noise_sigma=[[1. 0.], [0. 1.]] num_support_pts=5 kernel=RBFKernel(sigma=2)",
        # "KMPPI_K=500 T=20 M=1 lambda=10 noise_mu=[0. 0.] noise_sigma=[[1. 0.], [0. 1.]] num_support_pts=5 kernel=RBFKernel(sigma=2)",
        # "KMPPI_K=500 T=20 M=1 lambda=1 noise_mu=[0. 0.] noise_sigma=[[1. 0.], [0. 1.]] num_support_pts=5 kernel=RBFKernel(sigma=3)",
    ]

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 14))
    ax[0].set_xlim(0, num_steps - 1)
    # only set the min of y to be 0
    ax[0].set_title(f"trajectory cost")
    ax[1].set_title(f"rollout cost")
    ax[1].set_yscale('log')
    ax[1].set_xticks(range(num_steps))
    ax[1].set_xlabel("step")
    f, a = plt.subplots()
    a.set_title(f"control inputs total diff")
    # tick on x for every step discretely
    for method in allowed_names:
        data = methods[method]
        method = simplify_name(method)
        actual_costs = torch.stack(data["actual_costs"])
        rollout_costs = torch.stack(data["rollout_costs"])
        controls = data["controls"]
        control_diff = data["control_diff"]
        for i, v in enumerate([actual_costs, rollout_costs]):
            # plot the median along dim 0 and the 25th and 75th percentile
            lower = torch.quantile(v, .25, dim=0)
            upper = torch.quantile(v, .75, dim=0)
            ax[i].fill_between(range(num_steps), lower, upper, alpha=0.2)
            ax[i].plot(v.median(dim=0)[0], label=method)

        # compute total control diff
        control_diff = torch.stack(control_diff)
        total_diff = control_diff.abs().sum(dim=(1, 2))
        c1 = actual_costs.sum(dim=1)
        c2 = rollout_costs.sum(dim=1)
        print(
            f"method {method}\ntrajectory cost {c1.mean():.1f} ({c1.std():.1f})\nrollout cost {c2.mean():.1f} ({c2.std():.1f})\ncontrol diff {total_diff.mean():.1f} ({total_diff.std():.1f})")
        # plot frequency of total control diff
        # kernel density estimate
        from scipy.stats import gaussian_kde
        density = gaussian_kde(total_diff)
        density.covariance_factor = lambda: .25
        density._compute_covariance()
        xs = torch.linspace(0, total_diff.max() * 1.2, 50)
        a.plot(xs, density(xs), label=method)
        # plot histogram of total control diff

    ax[0].set_ylim(0, None)
    ax[1].legend()
    a.set_ylim(0, None)
    a.set_xlim(0, None)
    a.legend()
    plt.show()
    plt.tight_layout()
    input("Press Enter to close the window and exit...")


def main(plot_only=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.double
    pytorch_seed.seed(2)
    ch = cache.LocalCache("mppi_res.pkl")

    if plot_only:
        plot_result(ch)
        return

    # create toy environment to do on control on (default start and goal)
    env = Toy2DEnvironment(visualize=True, terminal_scale=10, device=device)
    shared_params = {
        "num_samples": 500,
        "horizon": 20,
        "noise_mu": torch.zeros(2, dtype=dtype, device=device),
        "noise_sigma": torch.diag(torch.tensor([1., 1.], dtype=dtype, device=device)),
        "u_max": torch.tensor([1., 1.], dtype=dtype, device=device),
        "terminal_state_cost": env.terminal_cost,
        "lambda_": 1,
        "device": device,
    }
    # create MPPI with some initial parameters
    mmppi = mppi.MPPI(env.dynamics, env.running_cost, 2,
                      **shared_params)
    smppi = mppi.SMPPI(env.dynamics, env.running_cost, 2,
                       **shared_params,
                       w_action_seq_cost=10,
                       action_max=torch.tensor([1., 1.], dtype=dtype, device=device))
    kmppi = mppi.KMPPI(env.dynamics, env.running_cost, 2,
                       **shared_params,
                       kernel=mppi.RBFKernel(sigma=2),
                       num_support_pts=5,
                       )

    for ctrl in [mmppi, kmppi, smppi]:
        do_control(env, ctrl, ch, seeds=range(5), run_steps=20, num_refinement_steps=1, save_img=True,
                   plot_single=False, evaluate_running_cost=False, plot_trajectory_candidates=True)


if __name__ == "__main__":
    main()
