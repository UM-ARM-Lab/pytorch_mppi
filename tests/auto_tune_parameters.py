import torch
import typing

import window_recorder
from arm_pytorch_utilities import linalg
import matplotlib.colors
from matplotlib import pyplot as plt

from pytorch_mppi.mppi import handle_batch_input

from pytorch_mppi import autotune

from pytorch_mppi import MPPI
from pytorch_seed import seed
import logging
# import window_recorder
from contextlib import nullcontext

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
        self.start_visualization()

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
            self.draw_start()
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
            r = torch.cat((self.start.reshape(1, -1), rollout))
            artists += self.ax.plot(r[:, 0], r[:, 1], color="skyblue")
            artists += [self.ax.scatter(r[-1, 0], r[-1, 1], color="tab:red")]
        self.rollout_artist = artists
        plt.pause(0.001)

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
    env = Toy2DEnvironment(visualize=True, terminal_scale=10)

    # create MPPI with some initial parameters
    mppi = MPPI(env.dynamics, env.running_cost, 2,
                noise_sigma=torch.diag(torch.tensor([5., 5.], dtype=dtype, device=device)),
                num_samples=500,
                horizon=20, device=device,
                terminal_state_cost=env.terminal_cost,
                u_max=torch.tensor([2., 2.], dtype=dtype, device=device),
                lambda_=1)

    # use the same nominal trajectory to start with for all the evaluations for fairness
    nominal_trajectory = mppi.U.clone()
    # parameters for our sample evaluation function - lots of choices for the evaluation function
    evaluate_running_cost = True
    num_refinement_steps = 10
    num_trajectories = 5

    def evaluate():
        costs = []
        rollouts = []
        # we sample multiple trajectories for the same start to goal problem, but in your case you should consider
        # evaluating over a diverse dataset of trajectories
        for j in range(num_trajectories):
            mppi.U = nominal_trajectory.clone()
            # the nominal trajectory at the start will be different if the horizon's changed
            mppi.change_horizon(mppi.T)
            # usually MPPI will have its nominal trajectory warm-started from the previous iteration
            # for a fair test of tuning we will reset its nominal trajectory to the same random one each time
            # we manually warm it by refining it for some steps
            for k in range(num_refinement_steps):
                mppi.command(env.start, shift_nominal_trajectory=False)

            rollout = mppi.get_rollouts(env.start)

            this_cost = 0
            rollout = rollout[0]
            # here we evaluate on the rollout MPPI cost of the resulting trajectories
            # alternative costs for tuning the parameters are possible, such as just considering terminal cost
            if evaluate_running_cost:
                for t in range(len(rollout) - 1):
                    this_cost = this_cost + env.running_cost(rollout[t], mppi.U[t])
            this_cost = this_cost + env.terminal_cost(rollout, mppi.U)

            rollouts.append(rollout)
            costs.append(this_cost)
        return autotune.EvaluationResult(torch.stack(costs), torch.stack(rollouts))

    # choose from autotune.AutotuneMPPI.TUNABLE_PARAMS
    params_to_tune = [autotune.SigmaParameter(mppi), autotune.HorizonParameter(mppi), autotune.LambdaParameter(mppi)]
    # create a tuner with a CMA-ES optimizer
    # tuner = autotune.Autotune(params_to_tune, evaluate_fn=evaluate, optimizer=autotune.CMAESOpt(sigma=1.0))
    # # tune parameters for a number of iterations
    # with window_recorder.WindowRecorder(["Figure 1"]):
    #     iterations = 30
    #     for i in range(iterations):
    #         # results of this optimization step are returned
    #         res = tuner.optimize_step()
    #         # we can render the rollouts in the environment
    #         env.draw_rollouts(res.rollouts)
    #
    # # get best results and apply it to the controller
    # # (by default the controller will take on the latest tuned parameter, which may not be best)
    # res = tuner.get_best_result()
    # tuner.apply_parameters(res.params)
    # env.draw_results(res.params, tuner.results)

    try:
        # can also use a Ray Tune optimizer, see
        # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#search-algorithms-tune-search
        # rather than adapting the current parameters, these optimizers allow you to define a search space for each
        # and will search on that space
        # be sure to close plt windows or else ray will duplicate them
        from pytorch_mppi import autotune_global
        from ray.tune.search.hyperopt import HyperOptSearch
        from ray.tune.search.bayesopt import BayesOptSearch

        params_to_tune = [autotune_global.SigmaGlobalParameter(mppi),
                          autotune_global.HorizonGlobalParameter(mppi),
                          autotune_global.LambdaGlobalParameter(mppi)]
        env.visualize = False
        plt.close('all')
        tuner = autotune_global.AutotuneGlobal(params_to_tune, evaluate_fn=evaluate,
                                               optimizer=autotune_global.RayOptimizer(HyperOptSearch))
        # ray tuners cannot be tuned iteratively, but you can specify how many iterations to tune for
        res = tuner.optimize_all(100)
        env.visualize = True
        env.start_visualization()
        env.draw_rollouts(res.rollouts)
        env.draw_results(res.params, tuner.results)

        # can also use quality diversity optimization
        # import pytorch_mppi.autotune_qd
        # optim = pytorch_mppi.autotune_qd.CMAMEOpt()
        # tuner = autotune_global.AutotuneGlobal(mppi, params_to_tune, evaluate_fn=evaluate,
        #                                        optimizer=optim)
        #
        # iterations = 10
        # for i in range(iterations):
        #     # results of this optimization step are returned
        #     res = tuner.optimize_step()
        #     # we can render the rollouts in the environment
        #     best_params = optim.get_diverse_top_parameters(5)
        #     for res in best_params:
        #         logger.info(res)

    except ImportError:
        print("To test the ray tuning, install with:\npip install 'ray[tune]' bayesian-optimization hyperopt")
        pass


if __name__ == "__main__":
    main()
