import torch
import time
import logging
from torch.distributions.multivariate_normal import MultivariateNormal

logger = logging.getLogger(__name__)


class MPPI():
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    def __init__(self, dynamics, nx, K, T, running_cost, device="cpu", terminal_state_cost=None,
                 lambda_=1.,
                 noise_mu=torch.tensor(0., dtype=torch.double),
                 noise_sigma=torch.tensor(1., dtype=torch.double),
                 u_init=torch.tensor(1., dtype=torch.double),
                 U_init=None):
        self.d = device
        self.dtype = u_init.dtype
        self.K = K  # N_SAMPLES
        self.T = T  # TIMESTEPS

        # dimensions of state and control
        self.nx = nx
        self.nu = 1 if len(u_init.shape) is 0 else u_init.shape[0]
        self.lambda_ = lambda_

        # handle 1D edge case
        if self.nu is 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
        # T x nu control sequence
        self.U = U_init
        self.u_init = u_init

        if self.U is None:
            self.U = self.noise_dist.sample((self.T,))

        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.state = None

    def _start_action_consideration(self):
        # reseample noise each time we take an action; these can be done at the start
        self.noise = self.noise_dist.sample((self.K, self.T))
        # cache action cost
        self.action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv

    def _compute_total_cost_batch(self):
        # parallelize sampling across trajectories
        cost_total = torch.zeros(self.K, device=self.d, dtype=self.dtype)
        state = self.state.view(1, -1).repeat(self.K, 1)
        # broadcast own control to noise over samples; now it's K x T x nu
        perturbed_action = self.U + self.noise
        for t in range(self.T):
            u = perturbed_action[:, t]
            state = self.F(state, u)
            cost_total += self.running_cost(state, u)
        # action perturbation cost
        perturbation_cost = torch.sum(perturbed_action * self.action_cost, dim=(1, 2))
        if self.terminal_state_cost:
            cost_total += self.terminal_state_cost(state)
        cost_total += perturbation_cost
        return cost_total

    def _ensure_non_zero(self, cost, beta, factor):
        return torch.exp(-factor * (cost - beta))

    def command(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)

        self._start_action_consideration()
        cost_total = self._compute_total_cost_batch()

        beta = torch.min(cost_total)
        cost_total_non_zero = self._ensure_non_zero(cost_total, beta, 1 / self.lambda_)

        eta = torch.sum(cost_total_non_zero)
        omega = (1. / eta) * cost_total_non_zero
        for t in range(self.T):
            self.U[t] += torch.sum(omega.view(-1, 1) * self.noise[:, t], dim=0)
        action = self.U[0]

        # shift command 1 time step
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init

        return action


def run_mppi(mppi, env, retrain_dynamics, retrain_after_iter=50, iter=1000):
    dataset = torch.zeros((retrain_after_iter, mppi.nx + mppi.nu), dtype=mppi.U.dtype, device=mppi.d)
    for i in range(iter):
        state = env.state.copy()
        command_start = time.perf_counter()
        action = mppi.command(state)
        elapsed = time.perf_counter() - command_start
        s, r, _, _ = env.step(action.numpy())
        logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -r, elapsed)
        env.render()

        di = i % retrain_after_iter
        if di == 0 and i > 0:
            retrain_dynamics(dataset)
            # don't have to clear dataset since it'll be overridden, but useful for debugging
            dataset.zero_()
        dataset[di, :mppi.nx] = torch.tensor(state, dtype=mppi.U.dtype)
        dataset[di, mppi.nx:] = action
