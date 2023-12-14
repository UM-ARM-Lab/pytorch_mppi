import abc
import numpy as np
import torch.cuda

# pip install "ray[tune]" bayesian-optimization hyperopt
from ray import tune
from ray import train

from pytorch_mppi import autotune
from ray.tune.search.hyperopt import HyperOptSearch


class GlobalTunableParameter(autotune.TunableParameter, abc.ABC):
    def __init__(self, search_space):
        self.search_space = search_space

    @abc.abstractmethod
    def total_search_space(self) -> dict:
        """Return the potentially multidimensional search space for this parameter, which is a dictionary mapping
        each of the parameter's corresponding config names to a search space."""

    def get_linearized_search_space_value(self, param_values):
        if self.dim() == 1:
            return [self._linearize_space_value(self.search_space, param_values[self.name()])]
        return [self._linearize_space_value(self.search_space, param_values[f"{self.name()}"][i].item()) for i in
                range(self.dim())]

    @staticmethod
    def linearize_search_space(space):
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


class SigmaGlobalParameter(autotune.SigmaParameter, GlobalTunableParameter):
    def __init__(self, *args, search_space=tune.loguniform(1e-4, 1e2), **kwargs):
        super().__init__(*args, **kwargs)
        GlobalTunableParameter.__init__(self, search_space)

    def total_search_space(self) -> dict:
        return {f"{self.name()}{i}": self.search_space for i in range(self.dim())}


class MuGlobalParameter(autotune.MuParameter, GlobalTunableParameter):
    def __init__(self, *args, search_space=tune.uniform(-1, 1), **kwargs):
        super().__init__(*args, **kwargs)
        GlobalTunableParameter.__init__(self, search_space)

    def total_search_space(self) -> dict:
        return {f"{self.name()}{i}": self.search_space for i in range(self.dim())}


class LambdaGlobalParameter(autotune.LambdaParameter, GlobalTunableParameter):
    def __init__(self, *args, search_space=tune.loguniform(1e-5, 1e3), **kwargs):
        super().__init__(*args, **kwargs)
        GlobalTunableParameter.__init__(self, search_space)

    def total_search_space(self) -> dict:
        return {self.name(): self.search_space}


class HorizonGlobalParameter(autotune.HorizonParameter, GlobalTunableParameter):
    def __init__(self, *args, search_space=tune.randint(1, 50), **kwargs):
        super().__init__(*args, **kwargs)
        GlobalTunableParameter.__init__(self, search_space)

    def total_search_space(self) -> dict:
        return {self.name(): self.search_space}


class AutotuneGlobal(autotune.Autotune):
    def search_space(self):
        space = {}
        for p in self.params:
            assert isinstance(p, GlobalTunableParameter)
            space.update(p.total_search_space())
        return space

    def linearized_search_space(self):
        return {k: GlobalTunableParameter.linearize_search_space(space) for k, space in self.search_space().items()}

    def linearize_params(self, param_values):
        v = []
        for p in self.params:
            assert isinstance(p, GlobalTunableParameter)
            v.extend(p.get_linearized_search_space_value(param_values))
        return np.array(v)

    def initial_value(self):
        init = {}
        param_values = self.get_parameter_values(self.params)
        for p in self.params:
            assert isinstance(p, GlobalTunableParameter)
            init.update(p.get_config_from_parameter_value(param_values[p.name()]))
        return init


class RayOptimizer(autotune.Optimizer):
    def __init__(self, search_alg=HyperOptSearch,
                 default_iterations=100):
        self.iterations = default_iterations
        self.search_alg = search_alg
        self.all_res = None
        super().__init__()

    def setup_optimization(self):
        if not isinstance(self.tuner, AutotuneGlobal):
            raise RuntimeError(f"Ray optimizers require global search space information provided by AutotuneMPPIGlobal")
        space = self.tuner.search_space()
        init = self.tuner.initial_value()

        hyperopt_search = self.search_alg(points_to_evaluate=[init], metric="cost", mode="min")

        trainable_with_resources = tune.with_resources(self.trainable, {"gpu": 1 if torch.cuda.is_available() else 0})
        self.optim = tune.Tuner(
            trainable_with_resources,
            tune_config=tune.TuneConfig(
                num_samples=self.iterations,
                search_alg=hyperopt_search,
                metric="cost",
                mode="min",
            ),
            param_space=space,
        )

    def trainable(self, config):
        self.tuner.attach_parameters()
        self.tuner.apply_parameters(self.tuner.config_to_params(config))
        res = self.tuner.evaluate_fn()
        train.report({'cost': res.costs.mean().item()})

    def optimize_step(self):
        raise RuntimeError("Ray optimizers only allow tuning of all iterations at once")

    def optimize_all(self, iterations):
        self.iterations = iterations
        self.setup_optimization()
        self.all_res = self.optim.fit()
        self.tuner.apply_parameters(self.tuner.config_to_params(self.all_res.get_best_result().config))
        res = self.tuner.evaluate_fn()
        return res
