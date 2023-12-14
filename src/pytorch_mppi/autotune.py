import logging
import abc

import numpy as np
import torch
import typing

from arm_pytorch_utilities.tensor_utils import ensure_tensor
from torch.distributions import MultivariateNormal

from pytorch_mppi import MPPI
# optimizers
import cma

logger = logging.getLogger(__file__)


class EvaluationResult(typing.NamedTuple):
    # (N) cost for each trajectory evaluated
    costs: torch.Tensor
    # (N x H x nx) where H is the horizon and nx is the state dimension
    rollouts: torch.Tensor
    # parameter values populated by the tuner after evaluation returns
    params: dict = None
    # iteration number populated by the tuner after evaluation returns
    iteration: int = None


class Optimizer:
    def __init__(self):
        self.tuner: typing.Optional[Autotune] = None
        self.optim = None

    @abc.abstractmethod
    def setup_optimization(self) -> None:
        """Create backend optim object with optimization parameters and MPPI parameters from the tuner"""

    @abc.abstractmethod
    def optimize_step(self) -> EvaluationResult:
        """Optimize a single step, returning the evaluation result from the latest parameters"""

    def optimize_all(self, iterations) -> EvaluationResult:
        """Optimize multiple steps, returning the best evaluation results.
        Some optimizers may only have this implemented."""
        res = None
        for i in range(iterations):
            res = self.optimize_step()
        return res


class CMAESOpt(Optimizer):
    """Optimize using CMA-ES, an evolutionary algorithm that maintains a Gaussian population,
    starting around the initial parameters with a variance (potentially different for each hyperparameter)."""

    def __init__(self, population=10, sigma=0.1):
        self.population = population
        self.sigma = sigma
        super().__init__()

    def setup_optimization(self):
        x0 = self.tuner.flatten_params()

        options = {"popsize": self.population, "seed": np.random.randint(0, 10000), "tolfun": 1e-5, "tolfunhist": 1e-6}
        self.optim = cma.CMAEvolutionStrategy(x0=x0, sigma0=self.sigma, inopts=options)

    def optimize_step(self):
        params = self.optim.ask()
        # convert params for use

        cost_per_param = []
        all_rollouts = []
        for param in params:
            self.tuner.unflatten_params(param)
            res = self.tuner.evaluate_fn()
            cost_per_param.append(res.costs.mean().cpu().numpy())
            all_rollouts.append(res.rollouts)

        cost_per_param = np.array(cost_per_param)
        self.optim.tell(params, cost_per_param)

        best_param = self.optim.best.x
        self.tuner.unflatten_params(best_param)
        res = self.tuner.evaluate_fn()
        return res


class TunableParameter(abc.ABC):
    """A parameter that can be tuned by the autotuner. Holds references to the object that defines its actual value."""

    @staticmethod
    @abc.abstractmethod
    def name():
        """Get the name of the parameter"""

    @abc.abstractmethod
    def dim(self):
        """Get the dimension of the parameter"""

    @abc.abstractmethod
    def get_current_parameter_value(self):
        """Get the current underlying value of the parameter"""

    @abc.abstractmethod
    def ensure_valid_value(self, value):
        """Return a validated parameter value as close in intent as the input value as possible"""

    @abc.abstractmethod
    def apply_parameter_value(self, value):
        """Apply the parameter value to the underlying object"""

    @abc.abstractmethod
    def attach_to_state(self, state: dict):
        """Reattach/reinitialize the parameter to a new internal state. This should be similar to a call to __init__"""

    def get_parameter_value_from_config(self, config):
        """Get the serialized value of the parameter from a config dictionary, where each name is a scalar"""
        return config[self.name()]

    def get_config_from_parameter_value(self, value):
        """Reverse of the above method, get a config dictionary from a parameter value"""
        return {self.name(): value}


class MPPIParameter(TunableParameter, abc.ABC):
    def __init__(self, mppi: MPPI, dim=None):
        self.mppi = mppi
        self._dim = dim
        if self.mppi is not None:
            self.d = self.mppi.d
            self.dtype = self.mppi.dtype
            if dim is None:
                self._dim = self.mppi.nu

    def attach_to_state(self, state: dict):
        self.mppi = state['mppi']
        self.d = self.mppi.d
        self.dtype = self.mppi.dtype


class SigmaParameter(MPPIParameter):
    eps = 0.0001

    @staticmethod
    def name():
        return 'sigma'

    def dim(self):
        return self._dim

    def get_current_parameter_value(self):
        return torch.cat([self.mppi.noise_sigma[i][i].view(1) for i in range(self.dim())])

    def ensure_valid_value(self, value):
        sigma = ensure_tensor(self.d, self.dtype, value)
        sigma[sigma < self.eps] = self.eps
        return sigma

    def apply_parameter_value(self, value):
        sigma = self.ensure_valid_value(value)
        self.mppi.noise_sigma = torch.diag(sigma)
        self.mppi.noise_dist = MultivariateNormal(self.mppi.noise_mu, covariance_matrix=self.mppi.noise_sigma)
        self.mppi.noise_sigma_inv = torch.inverse(self.mppi.noise_sigma.detach())

    def get_parameter_value_from_config(self, config):
        return torch.tensor([config[f'{self.name()}{i}'] for i in range(self.dim())], dtype=self.dtype, device=self.d)

    def get_config_from_parameter_value(self, value):
        return {f'{self.name()}{i}': value[i].item() for i in range(self.dim())}


class MuParameter(MPPIParameter):
    @staticmethod
    def name():
        return 'mu'

    def dim(self):
        return self._dim

    def get_current_parameter_value(self):
        return self.mppi.noise_mu.clone()

    def ensure_valid_value(self, value):
        mu = ensure_tensor(self.d, self.dtype, value)
        return mu

    def apply_parameter_value(self, value):
        mu = self.ensure_valid_value(value)
        self.mppi.noise_dist = MultivariateNormal(mu, covariance_matrix=self.mppi.noise_sigma)
        self.mppi.noise_sigma_inv = torch.inverse(self.mppi.noise_sigma.detach())

    def get_parameter_value_from_config(self, config):
        return torch.tensor([config[f'{self.name()}{i}'] for i in range(self.dim())], dtype=self.dtype, device=self.d)

    def get_config_from_parameter_value(self, value):
        return {f'{self.name()}{i}': value[i].item() for i in range(self.dim())}


class LambdaParameter(MPPIParameter):
    eps = 0.0001

    @staticmethod
    def name():
        return 'lambda'

    def dim(self):
        return 1

    def get_current_parameter_value(self):
        return self.mppi.lambda_

    def ensure_valid_value(self, value):
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            value = value[0]
        v = max(value, self.eps)
        return v

    def apply_parameter_value(self, value):
        v = self.ensure_valid_value(value)
        self.mppi.lambda_ = v


class HorizonParameter(MPPIParameter):
    @staticmethod
    def name():
        return 'horizon'

    def dim(self):
        return 1

    def get_current_parameter_value(self):
        return self.mppi.T

    def ensure_valid_value(self, value):
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            value = value[0]
        v = max(round(value), 1)
        return v

    def apply_parameter_value(self, value):
        v = self.ensure_valid_value(value)
        self.mppi.change_horizon(v)


class Autotune:
    """Tune selected hyperparameters using state-of-the-art optimizers on an evaluation function.
    Subclass to define other parameters to optimize over such as terminal cost scaling. 
    See tests/auto_tune_parameters.py for an example evaluate_fn
    """
    eps = 0.0001

    def __init__(self, params_to_tune: typing.Sequence[TunableParameter],
                 evaluate_fn: typing.Callable[[], EvaluationResult],
                 reload_state_fn: typing.Callable[[], dict] = None,
                 optimizer=CMAESOpt()):
        """

        :param params_to_tune: sequence of tunable parameters
        :param evaluate_fn: function that returns an EvaluationResult that we want to minimize
        :param reload_state_fn: function that returns a dictionary of state to reattach to the parameters
        :param optimizer: optimizer that searches in the parameter space
        """
        self.evaluate_fn = evaluate_fn
        self.reload_state_fn = reload_state_fn

        self.params = params_to_tune
        self.optim = optimizer
        self.optim.tuner = self
        self.results = []

        self.attach_parameters()
        self.optim.setup_optimization()

    def optimize_step(self) -> EvaluationResult:
        res = self.optim.optimize_step()
        res = self.log_current_result(res)
        return res

    def optimize_all(self, iterations) -> EvaluationResult:
        res = self.optim.optimize_all(iterations)
        res = self.log_current_result(res)
        return res

    def get_best_result(self) -> EvaluationResult:
        return min(self.results, key=lambda res: res.costs.mean().item())

    def log_current_result(self, res: EvaluationResult):
        with torch.no_grad():
            iteration = len(self.results)
            kv = self.get_parameter_values(self.params)
            res = res._replace(iteration=iteration,
                               params={k: v.detach().clone() if torch.is_tensor(v) else v for k, v in
                                       kv.items()})
            logger.info(f"i:{iteration} cost: {res.costs.mean().item()} params:{kv}")
            self.results.append(res)
        return res

    def get_parameter_values(self, params_to_tune: typing.Sequence[TunableParameter]):
        # take on the assigned values to the MPPI
        return {p.name(): p.get_current_parameter_value() for p in params_to_tune}

    def flatten_params(self):
        x = []
        kv = self.get_parameter_values(self.params)
        # TODO ensure this is the same order as define and unflatten
        for k, v in kv.items():
            if torch.is_tensor(v):
                x.append(v.detach().cpu().numpy())
            else:
                x.append([v])
        x = np.concatenate(x)
        return x

    def unflatten_params(self, x, apply=True):
        # have to be in the same order as the flattening
        param_values = {}
        i = 0
        for p in self.params:
            raw_value = x[i:i + p.dim()]
            param_values[p.name()] = p.ensure_valid_value(raw_value)
            i += p.dim()
        if apply:
            self.apply_parameters(param_values)
        return param_values

    def apply_parameters(self, param_values):
        for p in self.params:
            p.apply_parameter_value(param_values[p.name()])

    def attach_parameters(self):
        """Attach parameters to any underlying state they require In most cases the parameters are defined already
        attached to whatever state it needs, e.g. the MPPI controller object for changing the parameter values.
        However, there are cases where the full state is not serializable, e.g. when using a multiprocessing pool
        and so we pass only the information required to load the state. We then must load the state and reattach
        the parameters to the state each training iteration."""
        if self.reload_state_fn is not None:
            state = self.reload_state_fn()
            for p in self.params:
                p.attach_to_state(state)

    def config_to_params(self, config):
        """Configs are param dictionaries where each must be a scalar"""
        return {p.name(): p.get_parameter_value_from_config(config) for p in self.params}
