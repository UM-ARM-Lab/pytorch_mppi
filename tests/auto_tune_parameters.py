import torch
from torch.distributions import MultivariateNormal
from arm_pytorch_utilities import linalg
import matplotlib.colors
from matplotlib import pyplot as plt
from pytorch_mppi.mppi import handle_batch_input

from pytorch_mppi import MPPI
from pytorch_seed import seed
import logging
import window_recorder

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
    def __init__(self, start=None, goal=None, dtype=torch.double, device="cpu", r=0.01):
        self.d = device
        self.dtype = dtype
        self.state_ranges = [
            (-5, 5),
            (-5, 5)
        ]

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

        K = 500
        T = 20
        self.terminal_scale = torch.tensor([10], dtype=self.dtype, device=self.d)
        self.lamb = torch.tensor([1], dtype=self.dtype, device=self.d)
        self.sigma = torch.tensor([0.2, 0.2], dtype=self.dtype, device=self.d, requires_grad=True)
        self.mppi = MPPI(self.dynamics, self.running_cost, 2, noise_sigma=torch.diag(self.sigma), num_samples=K,
                         horizon=T, device=self.d,
                         terminal_state_cost=self.terminal_cost,
                         lambda_=self.lamb)
        # use fixed nominal trajectory
        self.nominal_trajectory = self.mppi.U.clone()
        self.params = {
            # 'terminal scale': self.terminal_scale,
            # 'lambda': self.lamb,
            # 'sigma inv': self.mppi.noise_sigma_inv
            'sigma': self.sigma
        }
        self.results = []
        for v in self.params.values():
            v.requires_grad = True
        self.optim = torch.optim.Adam(self.params.values(), lr=0.1)

    def plan(self, iteration):
        costs = None
        rollouts = []
        for j in range(5):
            self.mppi.U = self.nominal_trajectory.clone()
            self.mppi.command(self.start, shift_nominal_trajectory=False)

            # with torch.no_grad():
            rollout = self.mppi.get_rollouts(self.start)
            rollouts.append(rollout)

            # TODO experiment with different costs to learn on
            # this_cost = self.mppi.cost_total
            this_cost = self.terminal_cost(rollout[0], self.mppi.U)

            if costs is None:
                costs = this_cost
            else:
                costs = torch.cat((costs, this_cost))

        with torch.no_grad():
            rollouts = torch.cat(rollouts)
            self.draw_rollouts(rollouts)
            logger.info(f"i:{iteration} cost: {costs.mean().item()} params:{self.params}")
            self.results.append({
                'iteration': iteration,
                'cost': costs.mean().item(),
                'params': {k: v.detach().clone() for k, v in self.params.items()},
            })

        costs.mean().backward()
        self.optim.step()
        # to remain positive definite
        with torch.no_grad():
            self.sigma[self.sigma < 0] = 0.0001
        self.mppi.noise_dist = MultivariateNormal(self.mppi.noise_mu, covariance_matrix=torch.diag(self.sigma))
        # TODO see if this needs to be attached
        self.mppi.noise_sigma_inv = torch.inverse(self.mppi.noise_sigma.detach())

        self.optim.zero_grad()

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
        self.clear_artist(self.rollout_artist)
        artists = []
        for rollout in rollouts:
            r = torch.cat((self.start.reshape(1, -1), rollout))
            artists += self.ax.plot(r[:, 0], r[:, 1], color="skyblue")
            artists += [self.ax.scatter(r[-1, 0], r[-1, 1], color="tab:red")]
        self.rollout_artist = artists
        plt.pause(0.001)

    def draw_costs(self, resolution=0.05, value_padding=0):
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
        a.append(self.ax.contourf(x, z, v, norm=norm, cmap=self.cmap))
        a.append(self.ax.contour(x, z, v, colors='k', linestyles='dashed'))
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
        self.clear_artist(self.start_artist)
        self.start_artist = self.draw_state(self.start, "tab:blue", label='start')

    def draw_goal(self):
        self.clear_artist(self.goal_artist)
        self.goal_artist = self.draw_state(self.goal, "tab:green", label='goal')

    def draw_state(self, state, color, label=None, ox=-0.3, oy=0.3):
        artists = [self.ax.scatter(state[0].cpu(), state[1].cpu(), color=color)]
        if label is not None:
            artists.append(self.ax.text(state[0].cpu() + ox, state[1].cpu() + oy, label, color=color))
        plt.pause(0.0001)
        return artists


def main():
    seed(1)
    # torch.autograd.set_detect_anomaly(True)
    exp = Experiment()
    with window_recorder.WindowRecorder(["Figure 1"]):
        iterations = 100
        for i in range(iterations):
            exp.plan(i)
    exp.draw_results()

    input('finished')


if __name__ == "__main__":
    main()
