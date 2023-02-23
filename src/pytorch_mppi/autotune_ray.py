import torch

# pip install "ray[tune]" bayesian-optimization hyperopt
from ray import tune

from pytorch_mppi import autotune
from ray.tune.search.hyperopt import HyperOptSearch


class RayOptimizer(autotune.Optimizer):
    def __init__(self, search_alg=HyperOptSearch,
                 sigma_search_space=tune.loguniform(1e-4, 1e2),
                 mu_search_space=tune.uniform(-1, 1),
                 lambda_search_space=tune.loguniform(1e-5, 1e3),
                 horizon_search_space=tune.randint(1, 50),
                 default_iterations=100):
        self.iterations = default_iterations
        self.search_alg = search_alg
        self.sigma_search_space = sigma_search_space
        self.mu_search_space = mu_search_space
        self.lambda_search_space = lambda_search_space
        self.horizon_search_space = horizon_search_space
        self.all_res = None
        super().__init__()

    def setup_optimization(self):
        nu = self.tuner.mppi.nu
        p = self.tuner.params
        space = {}
        init = {}
        if 'sigma' in p:
            space.update({f"sigma{i}": self.sigma_search_space for i in range(nu)})
            init.update({f"sigma{i}": p['sigma'][i].item() for i in range(nu)})
        if 'mu' in p:
            space.update({f"mu{i}": self.mu_search_space for i in range(nu)})
            init.update({f"mu{i}": p['mu'][i].item() for i in range(nu)})
        if 'lambda' in p:
            space['lambda'] = self.lambda_search_space
            init['lambda'] = p['lambda']
        if 'horizon' in p:
            space['horizon'] = self.horizon_search_space
            init['horizon'] = p['horizon']

        hyperopt_search = self.search_alg(points_to_evaluate=[init], metric="cost", mode="min")
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

    def config_to_params(self, config):
        nu = self.tuner.mppi.nu
        p = self.tuner.params

        dtype = self.tuner.dtype
        device = self.tuner.d

        params = {}
        for name in ['sigma', 'mu']:
            if name in p:
                params[name] = torch.tensor([config[f'{name}{i}'] for i in range(nu)], dtype=dtype, device=device)
        for name in ['lambda', 'horizon']:
            if name in p:
                params[name] = config[name]

        return params

    def trainable(self, config):
        self.tuner.apply_parameters(self.config_to_params(config))
        res = self.tuner.evaluate_fn()
        tune.report(cost=res.costs.mean().item())

    def optimize_step(self):
        raise RuntimeError("Ray optimizers only allow tuning of all iterations at once")

    def optimize_all(self, iterations):
        self.iterations = iterations
        self.setup_optimization()
        self.all_res = self.optim.fit()
        self.tuner.apply_parameters(self.config_to_params(self.all_res.get_best_result().config))
        res = self.tuner.evaluate_fn()
        return res
