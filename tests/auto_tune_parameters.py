import torch
from arm_pytorch_utilities import linalg
import matplotlib.colors
from matplotlib import pyplot as plt
from pytorch_mppi.mppi import handle_batch_input

from pytorch_mppi import autotune
from pytorch_mppi import autotune_ray

from pytorch_mppi import MPPI
from pytorch_seed import seed
import logging
import window_recorder
from contextlib import nullcontext

from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch

plt.switch_backend('Qt5Agg')

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


class LinearDynamics:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    @handle_batch_input(n=2)
    def __call__(self, state, action):
        nx = self.A @ state + self.B @ action
        return nx


class LinearDeltaDynamics:
    def __init__(self, B):
        self.B = B

    @handle_batch_input(n=2)
    def __call__(self, state, action):
        nx = state + action @ self.B.transpose(0, 1)
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


class Experiment(autotune.AutotuneMPPI):
    def __init__(self, start=None, goal=None, dtype=torch.double, device="cpu", r=0.01, evaluate_running_cost=True,
                 visualize=True,
                 num_refinement_steps=5,
                 num_trajectories=5, num_samples=500, horizon=20,
                 **kwargs):
        self.d = device
        self.dtype = dtype
        self.state_ranges = [
            (-5, 5),
            (-5, 5)
        ]
        self.evaluate_running_cost = evaluate_running_cost
        self.num_trajectories = num_trajectories
        self.num_refinement_steps = num_refinement_steps
        self.visualize = visualize
        self.nx = 2

        B = torch.tensor([[0.5, 0], [0, -0.5]], device=self.d, dtype=self.dtype)
        self.dynamics = LinearDeltaDynamics(B)

        self.start = start or torch.tensor([-3, -2], device=self.d, dtype=self.dtype)
        self.goal = goal or torch.tensor([2, 2], device=self.d, dtype=self.dtype)

        self.costs = []

        eye = torch.eye(2, device=self.d, dtype=self.dtype)
        goal_cost = LQRCost(eye, eye * r, self.goal)
        self.costs.append(goal_cost)

        # for increasing difficulty, we add some "hills"
        self.costs.append(HillCost(torch.tensor([[0.1, 0.05], [0.05, 0.1]], device=self.d, dtype=self.dtype) * 1.5,
                                   torch.tensor([-0.5, -1], device=self.d, dtype=self.dtype), cost_at_center=60))

        if self.visualize:
            self.start_visualization()

        self.terminal_scale = torch.tensor([10], dtype=self.dtype, device=self.d)
        sigma = torch.tensor([5.2, 5.2], dtype=self.dtype, device=self.d)
        mppi = MPPI(self.dynamics, self.running_cost, 2, noise_sigma=torch.diag(sigma),
                    num_samples=num_samples,
                    horizon=horizon, device=self.d,
                    terminal_state_cost=self.terminal_cost,
                    lambda_=1)
        super().__init__(mppi, ['sigma'], self._create_evaluate(), **kwargs)
        # use fixed nominal trajectory
        self.nominal_trajectory = self.mppi.U.clone()

    def _create_evaluate(self, trajectory=None, num_runs=None):
        """Produce costs and rollouts from the current state of MPPI"""

        # cost is of the terminal cost of the rollout from MPPI running for 1 iteration, averaged over some trials
        # inheriting classes should change this to other evaluation methods
        def _evaluate():
            nonlocal trajectory, num_runs
            costs = []
            rollouts = []
            for j in range(self.num_trajectories):
                if trajectory is None:
                    trajectory = self.nominal_trajectory
                self.mppi.U = trajectory.clone()
                if num_runs is None:
                    num_runs = self.num_refinement_steps
                for k in range(num_runs):
                    self.mppi.command(self.start, shift_nominal_trajectory=False)

                # with torch.no_grad():
                rollout = self.mppi.get_rollouts(self.start)
                rollouts.append(rollout)

                this_cost = 0
                rollout = rollout[0]
                if self.evaluate_running_cost:
                    for t in range(len(rollout) - 1):
                        this_cost = this_cost + self.running_cost(rollout[t], self.mppi.U[t])
                this_cost = this_cost + self.terminal_cost(rollout, self.mppi.U)

                costs.append(this_cost)
            return autotune.EvaluationResult(torch.cat(costs), torch.cat(rollouts))

        return _evaluate

    def log_current_result(self, res):
        with torch.no_grad():
            super().log_current_result(res)
            self.draw_rollouts(res.rollouts)

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

    def draw_results(self):
        iterations = [res['iteration'] for res in self.results]
        loss = [res['cost'] for res in self.results]

        # loss curve
        fig, ax = plt.subplots()
        ax.plot(iterations, loss)
        ax.set_xlabel('iteration')
        ax.set_ylabel('cost')
        plt.pause(0.001)
        plt.savefig('cost.png')

        if 'sigma' in self.params:
            sigma = [res['params']['sigma'] for res in self.results]
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
        a.append(self.ax.contourf(x, z, v, levels=[2, 4, 8, 16, 24, 32, 40, 50, 60, 70, 80, 90, 100], norm=norm,
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
        if not self.visualize:
            return
        self.clear_artist(self.goal_artist)
        self.goal_artist = self.draw_state(self.goal, "tab:green", label='goal')

    def draw_state(self, state, color, label=None, ox=-0.3, oy=0.3):
        artists = [self.ax.scatter(state[0].cpu(), state[1].cpu(), color=color)]
        if label is not None:
            artists.append(self.ax.text(state[0].cpu() + ox, state[1].cpu() + oy, label, color=color))
        plt.pause(0.0001)
        return artists


class RayExperiment(Experiment):
    def __init__(self, *args, visualize=True, **kwargs):
        self._actually_visualize = visualize
        super().__init__(*args, visualize=False, **kwargs)

    def optimize_all(self, iterations):
        assert isinstance(self.optim, autotune_ray.RayOptimizer)
        self.optim.optimize_all(iterations)

        # avoid having too many plots pop up
        self.visualize = self._actually_visualize
        self.start_visualization()
        for iteration in range(iterations):
            res = self.optim.all_res.results[iteration]
            self.apply_parameters(self.optim.config_to_params(res.config))
            eval_res = self.evaluate_fn()
            self.log_current_result(eval_res)
        self.visualize = False

        # best parameters
        self.apply_parameters(self.optim.config_to_params(self.optim.all_res.get_best_result().config))
        best_res = self.evaluate_fn()
        self.log_current_result(best_res)
        return best_res


def main():
    seed(1)
    # torch.autograd.set_detect_anomaly(True)
    # exp = Experiment(visualize=False, num_refinement_steps=10, optimizer=autotune.CMAESOpt(sigma=1.0))
    # with window_recorder.WindowRecorder(["Figure 1"]) if exp.visualize else nullcontext():
    #     iterations = 50
    #     for i in range(iterations):
    #         exp.optimize_step()
    # exp = RayExperiment(visualize=True, num_refinement_steps=10, optimizer=BayesOptSearch)

    # exp = Experiment(visualize=False, num_refinement_steps=10, optimizer=autotune_ray.RayOptimizer(BayesOptSearch))
    exp = Experiment(visualize=False, num_refinement_steps=10, optimizer=autotune_ray.RayOptimizer(HyperOptSearch))
    exp.optimize_all(50)
    exp.draw_results()

    # input('finished')


if __name__ == "__main__":
    main()
