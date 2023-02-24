import numpy as np
import torch

# pip install "ray[tune]" bayesian-optimization hyperopt
from ray import tune

from pytorch_mppi import autotune
from ray.tune.search.hyperopt import HyperOptSearch


class AutotuneMPPIGlobal(autotune.AutotuneMPPI):
    def __init__(self, *args,
                 sigma_search_space=tune.loguniform(1e-4, 1e2),
                 mu_search_space=tune.uniform(-1, 1),
                 lambda_search_space=tune.loguniform(1e-5, 1e3),
                 horizon_search_space=tune.randint(1, 50),
                 **kwargs):
        self.sigma_search_space = sigma_search_space
        self.mu_search_space = mu_search_space
        self.lambda_search_space = lambda_search_space
        self.horizon_search_space = horizon_search_space
        super().__init__(*args, **kwargs)

    def search_space(self):
        nu = self.mppi.nu
        p = self.params
        space = {}
        if 'sigma' in p:
            space.update({f"sigma{i}": self.sigma_search_space for i in range(nu)})
        if 'mu' in p:
            space.update({f"mu{i}": self.mu_search_space for i in range(nu)})
        if 'lambda' in p:
            space['lambda'] = self.lambda_search_space
        if 'horizon' in p:
            space['horizon'] = self.horizon_search_space
        return space

    def linearized_search_space(self):
        return {k: self._linearize_search_space(space) for k, space in self.search_space().items()}

    def linearize_params(self, params):
        nu = self.mppi.nu
        p = params
        v = []
        if 'sigma' in p:
            v.extend([self._linearize_space_value(self.sigma_search_space, p['sigma'][i].item()) for i in range(nu)])
        if 'mu' in p:
            v.extend([self._linearize_space_value(self.mu_search_space, p['mu'][i].item()) for i in range(nu)])
        if 'lambda' in p:
            v.append(self._linearize_space_value(self.lambda_search_space, p['lambda']))
        if 'horizon' in p:
            v.append(self._linearize_space_value(self.horizon_search_space, p['horizon']))
        return torch.tensor(v, device=self.d, dtype=self.dtype)

    @staticmethod
    def _linearize_search_space(space):
        # tune doesn't have public API for type checking samplers
        sampler = space.get_sampler()
        if hasattr(sampler, 'base'):
            b = np.log(sampler.base)
            return np.log(space.lower) / b, np.log(space.upper) / b
        return space.lower, space.upper

    @staticmethod
    def _linearize_space_value(space, v):
        # tune doesn't have public API for type checking samplers
        sampler = space.get_sampler()
        # log
        if hasattr(sampler, 'base'):
            b = np.log(sampler.base)
            return np.log(v) / b
        # quantized
        if hasattr(sampler, 'q'):
            return np.round(np.divide(v, sampler.q)) * sampler.q
        return v

    def initial_value(self):
        nu = self.mppi.nu
        p = self.params
        init = {}
        if 'sigma' in p:
            init.update({f"sigma{i}": p['sigma'][i].item() for i in range(nu)})
        if 'mu' in p:
            init.update({f"mu{i}": p['mu'][i].item() for i in range(nu)})
        if 'lambda' in p:
            init['lambda'] = p['lambda']
        if 'horizon' in p:
            init['horizon'] = p['horizon']
        return init


class RayOptimizer(autotune.Optimizer):
    def __init__(self, search_alg=HyperOptSearch,
                 default_iterations=100):
        self.iterations = default_iterations
        self.search_alg = search_alg
        self.all_res = None
        super().__init__()

    def setup_optimization(self):
        if not isinstance(self.tuner, AutotuneMPPIGlobal):
            raise RuntimeError(f"Ray optimizers require global search space information provided by AutotuneMPPIGlobal")
        space = self.tuner.search_space()
        init = self.tuner.initial_value()

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

    def trainable(self, config):
        self.tuner.apply_parameters(self.tuner.config_to_params(config))
        res = self.tuner.evaluate_fn()
        tune.report(cost=res.costs.mean().item())

    def optimize_step(self):
        raise RuntimeError("Ray optimizers only allow tuning of all iterations at once")

    def optimize_all(self, iterations):
        self.iterations = iterations
        self.setup_optimization()
        self.all_res = self.optim.fit()
        self.tuner.apply_parameters(self.tuner.config_to_params(self.all_res.get_best_result().config))
        res = self.tuner.evaluate_fn()
        return res


