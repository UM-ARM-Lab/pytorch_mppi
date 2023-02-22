import torch
from arm_pytorch_utilities import linalg
import matplotlib.colors
from matplotlib import pyplot as plt

from pytorch_mppi import MPPI

plt.switch_backend('Qt5Agg')


class LinearDynamics:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def __call__(self, state, action):
        nx = self.A @ state + self.B @ action
        return nx


class LinearDeltaDynamics:
    def __init__(self, B):
        self.B = B

    def __call__(self, state, action):
        nx = state + self.B @ action
        return nx


class LQRCost:
    def __init__(self, Q, R, goal):
        self.Q = Q
        self.R = R
        self.goal = goal

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

    def __call__(self, state, action=None):
        dx = self.center - state
        d = linalg.batch_quadratic_product(dx, self.Q)
        c = self.cost_at_center * torch.exp(-d)
        return c


class Experiment:
    def __init__(self, start=None, goal=None, dtype=torch.double, device="cpu", r=0.1):
        self.d = device
        self.dtype = dtype
        self.state_ranges = [
            (-5, 5),
            (-5, 5)
        ]

        B = torch.tensor([[1, 0], [0, 1]], device=self.d, dtype=self.dtype)
        self.dynamics = LinearDeltaDynamics(B)

        self.start = start or torch.tensor([-3, -2], device=self.d, dtype=self.dtype)
        self.goal = goal or torch.tensor([2, 2], device=self.d, dtype=self.dtype)

        self.costs = []

        goal_cost = LQRCost(torch.eye(2, device=self.d, dtype=self.dtype),
                            torch.eye(2, device=self.d, dtype=self.dtype) * r, self.goal)
        self.costs.append(goal_cost)

        # for increasing difficulty, we add some "hills"
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
        self.draw_costs()
        self.draw_start()
        self.draw_goal()

    def running_cost(self, state, action=None):
        c = None
        for cost in self.costs:
            if c is None:
                c = cost(state, action)
            else:
                c += cost(state, action)
        return c

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
        self.start_artist = self.draw_state(self.start, "blue", label='start')

    def draw_goal(self):
        self.clear_artist(self.goal_artist)
        self.goal_artist = self.draw_state(self.goal, "green", label='goal')

    def draw_state(self, state, color, label=None, ox=-0.3, oy=0.3):
        artists = [self.ax.scatter(state[0].cpu(), state[1].cpu(), color=color)]
        if label is not None:
            artists.append(self.ax.text(state[0].cpu() + ox, state[1].cpu() + oy, label, color=color))
        plt.pause(0.0001)
        return artists


def main():
    exp = Experiment()
    print("finished")


if __name__ == "__main__":
    main()
