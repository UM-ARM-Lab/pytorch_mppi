import torch


class MPPI():
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    def __init__(self, dynamics, nx, K, T, U, running_cost, terminal_state_cost=None,
                 lambda_=torch.tensor(1., dtype=torch.double),
                 noise_mu=torch.tensor(0., dtype=torch.double),
                 noise_sigma=torch.tensor(1., dtype=torch.double),
                 u_init=torch.tensor(1., dtype=torch.double),
                 noise_gaussian=True):
        self.K = K  # N_SAMPLES
        self.T = T  # TIMESTEPS
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.U = U
        self.u_init = u_init
        self.noise_gaussian = noise_gaussian
        # dimensions of state and control
        self.nx = nx
        self.nu = u_init.shape[0]

        # TODO check sizes

        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.state = None

    def _start_action_consideration(self):
        # reseample noise each time we take an action; these can be done at the start
        self.cost_total = np.zeros(shape=(self.K))
        if self.noise_gaussian:
            self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T))
        else:
            self.noise = np.full(shape=(self.K, self.T), fill_value=0.9)
        # cache action cost
        self.action_cost = self.lambda_ * (1 / self.noise_sigma) * self.noise

    def _compute_total_cost(self, k):
        state = torch.from_numpy(self.state)
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
        return np.exp(-factor * (cost - beta))

    def control(self, env, retrain_dynamics, retrain_after_iter=50, iter=1000):
        self.state = np.array(env.env.state)
        dataset = np.zeros((retrain_after_iter, self.nx + self.nu))
        for i in range(iter):
            self._start_action_consideration()
            for k in range(self.K):
                self._compute_total_cost(k)

            beta = np.min(self.cost_total)  # minimum cost of all trajectories
            cost_total_non_zero = self._ensure_non_zero(cost=self.cost_total, beta=beta, factor=1 / self.lambda_)

            eta = np.sum(cost_total_non_zero)
            omega = 1 / eta * cost_total_non_zero

            self.U += [np.sum(omega * self.noise[:, t]) for t in range(self.T)]

            pre_action_state = self.state.copy()
            action = self.U[0]
            # env.env.state = self.state
            s, r, _, _ = env.step([self.U[0]])
            print("action taken: {:.2f} cost received: {:.2f}".format(self.U[0], -r))
            env.render()

            self.U = np.roll(self.U, -1)  # shift all elements to the left
            self.U[-1] = self.u_init  #
            self.state = np.array(env.env.state)
            self.state[0] = angle_normalize(self.state[0])
            logger.debug(self.state)

            di = i % retrain_after_iter
            if di == 0 and i > 0:
                retrain_dynamics(dataset)
                # don't have to clear dataset since it'll be overridden, but useful for debugging
                dataset = np.zeros((retrain_after_iter, self.nx + self.nu))
            dataset[di, :nx] = pre_action_state
            dataset[di, nx:] = action
