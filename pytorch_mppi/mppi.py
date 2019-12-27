import torch
from torch.distributions.multivariate_normal import MultivariateNormal


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
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(noise_mu.to(self.d), covariance_matrix=noise_sigma.to(self.d))
        # T x nu control sequence
        self.U = U_init
        self.u_init = u_init
        # dimensions of state and control
        self.nx = nx
        self.nu = u_init.shape[0]

        # TODO check sizes

        if self.U is None:
            self.U = self.noise_dist.sample((self.T))

        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.state = None

    def _start_action_consideration(self):
        # reseample noise each time we take an action; these can be done at the start
        self.cost_total = torch.zeros(self.K, device=self.d)
        self.noise = self.noise_dist.sample((self.K, self.T))
        # TODO handle matrix multiply (self.noise[:,:] * self.noise_sigma_inv)?
        # cache action cost
        self.action_cost = self.lambda_ * self.noise_sigma_inv * self.noise

    def _compute_total_cost(self, k):
        state = torch.from_numpy(self.state).to(self.d)
        for t in range(self.T):
            perturbed_action_t = self.U[t] + self.noise[k, t]
            state = self.F(state, perturbed_action_t)
            self.cost_total[k] += self.running_cost(state, perturbed_action_t)
            # add action perturbation cost
            self.cost_total[k] += perturbed_action_t * self.action_cost[k, t]
        # this is the additional terminal cost (running state cost at T already accounted for)
        if self.terminal_state_cost:
            self.cost_total[k] += self.terminal_state_cost(state)

    def _ensure_non_zero(self, cost, beta, factor):
        return torch.exp(-factor * (cost - beta))

    def command(self, obs):
        if torch.is_tensor(obs):
            obs = torch.from_numpy(obs)
        self.state = obs.to(self.d)

        self._start_action_consideration()
        # TODO easily parallelizable step
        for k in range(self.K):
            self._compute_total_cost(k)

        beta = torch.min(self.cost_total)
        cost_total_non_zero = self._ensure_non_zero(self.cost_total, beta, 1 / self.lambda_)

        eta = torch.sum(cost_total_non_zero)
        omega = (1. / eta) * cost_total_non_zero
        # TODO check dimensions
        self.U += [torch.sum(omega * self.noise[:, t]) for t in range(self.T)]
        action = self.U[0]

        # shift command to the left
        self.U[:-1, :] = self.U[1:, :]
        self.U[-1] = self.u_init

        return action
