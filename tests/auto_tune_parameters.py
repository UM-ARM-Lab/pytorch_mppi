import torch
from torch.distributions import MultivariateNormal
from arm_pytorch_utilities import linalg
from arm_pytorch_utilities.tensor_utils import ensure_tensor
import matplotlib.colors
from matplotlib import pyplot as plt
from pytorch_mppi.mppi import handle_batch_input

import cma
import numpy as np

from pytorch_mppi import MPPI
from pytorch_seed import seed
import logging
import window_recorder
from contextlib import nullcontext

from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
from ray import air, tune
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


class Experiment:
    def __init__(self, start=None, goal=None, dtype=torch.double, device="cpu", r=0.01, evaluate_running_cost=True,
                 visualize=True,
                 num_refinement_steps=5,
                 num_trajectories=5, num_samples=500, horizon=20):
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

        self.results = []
        self.terminal_scale = torch.tensor([10], dtype=self.dtype, device=self.d)
        self.lamb = torch.tensor([1], dtype=self.dtype, device=self.d)
        self.sigma = torch.tensor([0.2, 0.2], dtype=self.dtype, device=self.d, requires_grad=True)
        self.mppi = MPPI(self.dynamics, self.running_cost, 2, noise_sigma=torch.diag(self.sigma),
                         num_samples=num_samples,
                         horizon=horizon, device=self.d,
                         terminal_state_cost=self.terminal_cost,
                         lambda_=self.lamb)
        # use fixed nominal trajectory
        self.nominal_trajectory = self.mppi.U.clone()
        self.params = None
        self.optim = None
        self.define_parameters()
        self.setup_optimization()

    def define_parameters(self):
        self.params = {
            # 'terminal scale': self.terminal_scale,
            # 'lambda': self.lamb,
            # 'sigma inv': self.mppi.noise_sigma_inv
            'sigma': self.sigma
        }

    def setup_optimization(self):
        for v in self.params.values():
            v.requires_grad = True
        self.optim = torch.optim.Adam(self.params.values(), lr=0.1)

    def evaluate(self, trajectory=None, num_runs=None):
        """Produce costs and rollouts from the current state of MPPI"""
        # cost is of the terminal cost of the rollout from MPPI running for 1 iteration, averaged over some trials
        # inheriting classes should change this to other evaluation methods
        costs = None
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

            # TODO experiment with different costs to learn on
            # this_cost = self.mppi.cost_total
            this_cost = 0
            rollout = rollout[0]
            if self.evaluate_running_cost:
                for t in range(len(rollout) - 1):
                    this_cost = this_cost + self.running_cost(rollout[t], self.mppi.U[t])
            this_cost = this_cost + self.terminal_cost(rollout, self.mppi.U)

            if costs is None:
                costs = this_cost
            else:
                costs = torch.cat((costs, this_cost))
        return costs, rollouts

    def log_current_result(self, iteration, costs, rollouts):
        with torch.no_grad():
            rollouts = torch.cat(rollouts)
            self.draw_rollouts(rollouts)
            logger.info(f"i:{iteration} cost: {costs.mean().item()} params:{self.params}")
            self.results.append({
                'iteration': iteration,
                'cost': costs.mean().item(),
                'params': {k: v.detach().clone() for k, v in self.params.items()},
            })

    def optimize_parameters(self, iteration):
        costs, rollouts = self.evaluate()
        self.log_current_result(iteration, costs, rollouts)

        costs.mean().backward()
        self.optim.step()
        # to remain positive definite
        with torch.no_grad():
            self.sigma[self.sigma < 0] = 0.0001
        self.mppi.noise_dist = MultivariateNormal(self.mppi.noise_mu, covariance_matrix=torch.diag(self.sigma))
        # TODO see if this needs to be attached
        self.mppi.noise_sigma_inv = torch.inverse(self.mppi.noise_sigma.detach())

        self.optim.zero_grad()

    def apply_parameters(self, params):
        if 'sigma' in params:
            self.sigma = ensure_tensor(self.d, self.dtype, params['sigma'])
            # to remain positive definite
            self.sigma[self.sigma < 0] = 0.0001
            self.params['sigma'] = self.sigma
            self.mppi.noise_dist = MultivariateNormal(self.mppi.noise_mu, covariance_matrix=torch.diag(self.sigma))
            self.mppi.noise_sigma_inv = torch.inverse(self.mppi.noise_sigma.detach())

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


class FlattenExperiment(Experiment):
    def flatten_params(self):
        x = []
        # TODO implement for other parameters
        if 'sigma' in self.params:
            x.append(self.sigma.detach().cpu().numpy())
        x = np.concatenate(x)
        return x

    def unflatten_params(self, x):
        # have to be in the same order as the flattening
        params = {}
        i = 0
        if 'sigma' in self.params:
            params['sigma'] = x[i:i + 2]
            i += 2
        self.apply_parameters(params)


class CMAESExperiment(FlattenExperiment):
    def __init__(self, *args, population=10, optim_sigma=0.1, log_best=True, **kwargs):
        self.population = population
        self.optim_sigma = optim_sigma
        self.log_best = log_best
        super().__init__(*args, **kwargs)

    def setup_optimization(self):
        self.sigma.requires_grad = False
        # need to flatten our parameters to R^m
        x0 = self.flatten_params()

        options = {"popsize": self.population, "seed": np.random.randint(0, 10000), "tolfun": 1e-5, "tolfunhist": 1e-6}
        self.optim = cma.CMAEvolutionStrategy(x0=x0, sigma0=self.optim_sigma, inopts=options)

    def optimize_parameters(self, iteration):
        params = self.optim.ask()
        # convert params for use

        cost_per_param = []
        all_rollouts = []
        for param in params:
            self.unflatten_params(param)
            costs, rollouts = self.evaluate()
            cost_per_param.append(costs.mean().cpu().numpy())
            all_rollouts.append(rollouts)

        cost_per_param = np.array(cost_per_param)
        self.optim.tell(params, cost_per_param)

        # whether the best solution or the average solution should be logged
        if self.log_best:
            best_param = self.optim.best.x
            self.unflatten_params(best_param)
            best_cost, best_rollout = self.evaluate()
            self.log_current_result(iteration, best_cost, best_rollout)
        else:
            self.unflatten_params(np.stack(params).mean(axis=0))
            self.log_current_result(iteration, cost_per_param, all_rollouts[0])


class MPPI2Experiment(FlattenExperiment):
    def __init__(self, *args, population=10, sigma_sigma=0.1, lambda_=1., time_varying_parameters=False, **kwargs):
        self.population = population
        self.sigma_sigma = sigma_sigma
        self.optim_lambda = lambda_
        self.time_varying_parameters = time_varying_parameters
        self.cached_costs = None
        super().__init__(*args, **kwargs)

    def setup_optimization(self):
        self.sigma.requires_grad = False

        # default parameters becomes u_init mean
        x0 = torch.tensor(self.flatten_params(), dtype=self.dtype, device=self.d)
        params_sigma = self.flatten_params_sigma()
        p_min, p_max = self.flatten_params_min_max()
        self.optim = MPPI(self.mpc_dynamics, self.mpc_running_cost, self.mppi.T * self.mppi.nu, params_sigma,
                          num_samples=self.population,
                          horizon=self.num_refinement_steps if self.time_varying_parameters else 1,
                          device=self.d, lambda_=self.optim_lambda,
                          u_init=x0, U_init=x0.repeat(self.num_refinement_steps, 1),
                          u_min=p_min, u_max=p_max)

    def flatten_params_sigma(self):
        sigma = []
        # TODO implement for other parameters
        if 'sigma' in self.params:
            sigma.append(torch.tensor([self.sigma_sigma, self.sigma_sigma], device=self.d, dtype=self.dtype))
        sigma = torch.cat(sigma)
        return torch.diag(sigma)

    def flatten_params_min_max(self):
        mins = []
        maxs = []
        # TODO implement for other parameters
        # TODO compute parameter dimensionality
        np = 2
        if 'sigma' in self.params:
            mins.append(torch.tensor([0.0001 for _ in range(np)], device=self.d, dtype=self.dtype))
            maxs.append(torch.tensor([10e8 for _ in range(np)], device=self.d, dtype=self.dtype))
        mins = torch.cat(mins)
        maxs = torch.cat(maxs)
        return mins, maxs

    def mpc_dynamics(self, trajectory, params):
        new_trajectories = []
        self.cached_costs = []
        for i, traj in enumerate(trajectory):
            U = traj.reshape(-1, self.nx)
            self.unflatten_params(params[i])
            costs, _ = self.evaluate(U, num_runs=1 if self.time_varying_parameters else None)
            new_trajectories.append(self.mppi.U.clone().reshape(-1))
            self.cached_costs.append(costs.mean())
        self.cached_costs = torch.stack(self.cached_costs)
        new_trajectories = torch.stack(new_trajectories)
        return new_trajectories

    def mpc_running_cost(self, trajectory, params):
        c = self.cached_costs
        # avoid using stale cache
        self.cached_costs = None
        return c

    def optimize_parameters(self, iteration):
        state = self.nominal_trajectory.reshape(-1)
        self.optim.command(state, shift_nominal_trajectory=False)
        # action rollouts are the parameters used
        # "state" rollouts are the nominal trajectories
        last_params = self.optim.U[-1]
        self.unflatten_params(last_params)

        if self.time_varying_parameters:
            trajectory_rollouts = self.optim.get_rollouts(state)
            last_trajectory = trajectory_rollouts[0, -1]
            # only get 1 run since the nominal trajectory is already the product of many runs
            costs, rollouts = self.evaluate(last_trajectory.reshape(self.mppi.T, self.mppi.nu), num_runs=1)
        else:
            costs, rollouts = self.evaluate()

        self.log_current_result(iteration, costs, rollouts)


class HyperOptExperiment(Experiment):
    def __init__(self, *args, iterations=50, visualize=True, **kwargs):
        self.iterations = iterations
        self._actually_visualize = visualize
        super().__init__(*args, visualize=False, **kwargs)

    def setup_optimization(self):
        self.sigma.requires_grad = False

        space = {
            "sigma0": hp.uniform("sigma0", 0.0001, 10),
            "sigma1": hp.uniform("sigma1", 0.0001, 10),
        }
        initial_params = [
            {
                "sigma0": self.sigma[0].item(), "sigma1": self.sigma[1].item()
            }
        ]
        hyperopt_search = HyperOptSearch(space, points_to_evaluate=initial_params, metric="cost", mode="min")
        self.optim = tune.Tuner(
            self.trainable,
            tune_config=tune.TuneConfig(
                num_samples=self.iterations,
                search_alg=hyperopt_search,
                metric="cost",
                mode="min",
            )
        )

    def config_to_params(self, config):
        params = {}
        if 'sigma' in self.params:
            params['sigma'] = torch.tensor([config['sigma0'], config['sigma1']], dtype=self.dtype, device=self.d)
        return params

    def trainable(self, config):
        self.apply_parameters(self.config_to_params(config))
        costs, rollouts = self.evaluate()
        tune.report(cost=costs.mean().item())

    def optimize_all_iterations(self):
        results = self.optim.fit()

        # avoid having too many plots pop up
        self.visualize = self._actually_visualize
        self.start_visualization()
        for iteration in range(self.iterations):
            res = results[iteration]
            self.apply_parameters(self.config_to_params(res.config))
            costs, rollouts = self.evaluate()
            self.log_current_result(iteration, costs, rollouts)
        self.visualize = False

    def optimize_parameters(self, iteration):
        raise RuntimeError("optimize all iterations should be called instead")


class BayesOptExperiment(HyperOptExperiment):
    def setup_optimization(self):
        self.sigma.requires_grad = False

        space = {
            "sigma0": tune.uniform(0.0001, 10),
            "sigma1": tune.uniform(0.0001, 10),
        }
        initial_params = [
            {
                "sigma0": self.sigma[0].item(), "sigma1": self.sigma[1].item()
            }
        ]
        hyperopt_search = BayesOptSearch(points_to_evaluate=initial_params, metric="cost", mode="min")
        self.optim = tune.Tuner(
            self.trainable,
            tune_config=tune.TuneConfig(
                num_samples=self.iterations,
                search_alg=hyperopt_search,
                metric="cost",
                mode="min",
            ),
            param_space=space,
        )


def main():
    seed(1)
    # torch.autograd.set_detect_anomaly(True)
    # exp = CMAESExperiment(visualize=True, num_refinement_steps=10)
    # with window_recorder.WindowRecorder(["Figure 1"]) if exp.visualize else nullcontext():
    #     iterations = 50
    #     for i in range(iterations):
    #         exp.optimize_parameters(i)
    exp = BayesOptExperiment(visualize=True, num_refinement_steps=10)
    exp.optimize_all_iterations()
    exp.draw_results()

    # input('finished')


if __name__ == "__main__":
    main()
