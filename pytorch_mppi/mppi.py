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
        self.cost_total = torch.zeros(self.K, device=self.d)
        self.noise = self.noise_dist.sample((self.K, self.T))
        # cache action cost
        self.action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv

    def _compute_total_cost(self, k):
        state = self.state.clone()
        for t in range(self.T):
            perturbed_action_t = self.U[t] + self.noise[k, t]
            state = self.F(state, perturbed_action_t)
            self.cost_total[k] += self.running_cost(state, perturbed_action_t).item()
            # add action perturbation cost
            self.cost_total[k] += perturbed_action_t @ self.action_cost[k, t]
        # this is the additional terminal cost (running state cost at T already accounted for)
        if self.terminal_state_cost:
            self.cost_total[k] += self.terminal_state_cost(state)

    def _ensure_non_zero(self, cost, beta, factor):
        return torch.exp(-factor * (cost - beta))

    def command(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.U.dtype, device=self.d)

        self._start_action_consideration()
        # TODO easily parallelizable step
        for k in range(self.K):
            self._compute_total_cost(k)

        beta = torch.min(self.cost_total)
        cost_total_non_zero = self._ensure_non_zero(self.cost_total, beta, 1 / self.lambda_)

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
