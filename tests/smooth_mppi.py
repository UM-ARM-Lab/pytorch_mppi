import copy

import torch
import typing

import window_recorder
from arm_pytorch_utilities import linalg, handle_batch_input
import matplotlib.colors
from matplotlib import pyplot as plt

from pytorch_mppi import autotune

from pytorch_mppi import MPPI, SMPPI, KMPPI
from pytorch_seed import seed
import logging

# import window_recorder

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

    def draw_results(self, params, all_results: typing.Sequence[autotune.EvaluationResult]):
        iterations = [res.iteration for res in all_results]
        loss = [res.costs.mean().item() for res in all_results]

        # loss curve
        fig, ax = plt.subplots()
        ax.plot(iterations, loss)
        ax.set_xlabel('iteration')
        ax.set_ylabel('cost')
        plt.pause(0.001)
        plt.savefig('cost.png')

        if 'sigma' in params:
            sigma = [res.params['sigma'] for res in all_results]
            sigma = torch.stack(sigma)
            fig, ax = plt.subplots(nrows=2, sharex=True)
            ax[0].plot(iterations, sigma[:, 0])
            ax[1].plot(iterations, sigma[:, 1])
            ax[1].set_xlabel('iteration')
            ax[0].set_ylabel('sigma[0]')
            ax[1].set_ylabel('sigma[1]')
            plt.draw()
            plt.pause(0.005)
            plt.savefig('sigma.png')

    def draw_rollouts(self, rollouts):
        if not self.visualize:
            return
        self.clear_artist(self.rollout_artist)
        artists = []
        for rollout in rollouts:
            # r = torch.cat((self.start.reshape(1, -1), rollout))
            r = rollout.cpu()
            artists += [self.ax.scatter(r[0, 0], r[0, 1], color="tab:blue")]
            artists += self.ax.plot(r[:, 0], r[:, 1], color="skyblue")
            artists += [self.ax.scatter(r[-1, 0], r[-1, 1], color="tab:red")]
        self.rollout_artist = artists
        plt.pause(0.001)

    def draw_trajectory_step(self, prev_state, cur_state, color="tab:blue"):
        if not self.visualize:
            return
        if self.trajectory_artist is None:
            self.trajectory_artist = []
        artists = self.trajectory_artist
        artists += self.ax.plot([prev_state[0].cpu(), cur_state[0].cpu()],
                                [prev_state[1].cpu(), cur_state[1].cpu()], color=color)
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
        a.append(self.ax.contour(x, z, v, levels=a[0].levels, colors='k', linestyles='dashed'))
        a.append(self.ax.clabel(a[1], a[1].levels, inline=True, fontsize=13))
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


def main():
    seed(1)
    device = "cpu"
    dtype = torch.double

    # create toy environment to do on control on (default start and goal)
    env = Toy2DEnvironment(visualize=True, terminal_scale=10, device=device)

    # create MPPI with some initial parameters
    # mppi = MPPI(env.dynamics, env.running_cost, 2,
    #             noise_sigma=torch.diag(torch.tensor([1., 1.], dtype=dtype, device=device)),
    #             num_samples=500,
    #             horizon=20, device=device,
    #             terminal_state_cost=env.terminal_cost,
    #             u_max=torch.tensor([2., 2.], dtype=dtype, device=device),
    #             lambda_=1)
    # mppi = SMPPI(env.dynamics, env.running_cost, 2,
    #              noise_sigma=torch.diag(torch.tensor([1., 1.], dtype=dtype, device=device)),
    #              w_action_seq_cost=0,
    #              num_samples=500,
    #              horizon=20, device=device,
    #              terminal_state_cost=env.terminal_cost,
    #              u_max=torch.tensor([1., 1.], dtype=dtype, device=device),
    #              action_max=torch.tensor([1., 1.], dtype=dtype, device=device),
    #              lambda_=10)
    mppi = KMPPI(env.dynamics, env.running_cost, 2,
                 noise_sigma=torch.diag(torch.tensor([1., 1.], dtype=dtype, device=device)),
                 num_samples=500,
                 horizon=20, device=device,
                 num_support_pts=5,
                 terminal_state_cost=env.terminal_cost,
                 u_max=torch.tensor([2., 2.], dtype=dtype, device=device),
                 lambda_=1)

    # use the same nominal trajectory to start with for all the evaluations for fairness
    # parameters for our sample evaluation function - lots of choices for the evaluation function
    evaluate_running_cost = True
    run_steps = 20
    num_refinement_steps = 1

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
            u = mppi.command(state, shift_nominal_trajectory=last_refinement)

        rollout = mppi.get_rollouts(state)

        rollout_cost = 0
        this_cost = env.running_cost(state)
        rollout = rollout[0]
        # here we evaluate on the rollout MPPI cost of the resulting trajectories
        # alternative costs for tuning the parameters are possible, such as just considering terminal cost
        if evaluate_running_cost:
            for t in range(len(rollout) - 1):
                rollout_cost = rollout_cost + env.running_cost(rollout[t], mppi.U[t])
        rollout_cost = rollout_cost + env.terminal_cost(rollout, mppi.U)
        env.draw_rollouts([rollout])

        prev_state = copy.deepcopy(state)
        state = env.step(u)
        env.draw_trajectory_step(prev_state, state)

        print(f"step {i} state {state} current cost {this_cost} rollout cost {rollout_cost}")
        actual_costs.append(this_cost.cpu())
        rollout_costs.append(rollout_cost.cpu())
        controls.append(u.cpu())

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

    controls = torch.stack(controls)
    # consider total difference
    control_diff = torch.diff(controls, dim=0)

    ax[2].plot(controls[:, 0], label='u0')
    ax[2].plot(controls[:, 1], label='u1')
    ax[2].legend()
    ax[2].set_title(f"control inputs total diff: {control_diff.abs().sum()}")
    ax[2].set_xticks(range(run_steps))
    plt.tight_layout()
    plt.pause(0.001)

    print(f"total accumulated cost: {sum(actual_costs)}")
    print(f"total accumulated rollout cost: {sum(rollout_costs)}")

    input("Press Enter to close the window and exit...")
    pass


if __name__ == "__main__":
    main()
