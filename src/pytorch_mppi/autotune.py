import enum
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
        self.tuner: typing.Optional[AutotuneMPPI] = None
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


class AutotuneMPPI:
    """Tune selected MPPI hyperparameters using state-of-the-art optimizers on an evaluation function.
    Subclass to define other parameters to optimize over such as terminal cost scaling. An example
    evaluate_fn:


    """
    TUNABLE_PARAMS = ['sigma', 'mu', 'lambda', 'horizon']
    eps = 0.0001

    def __init__(self, mppi: MPPI, params_to_tune: typing.Sequence[str],
                 evaluate_fn: typing.Callable[[], EvaluationResult], optimizer=CMAESOpt()):
        self.mppi = mppi
        self.evaluate_fn = evaluate_fn
        self.d = mppi.d
        self.dtype = mppi.dtype

        self.params = None
        self.optim = optimizer
        self.optim.tuner = self
        self.results = []

        self.define_parameters(params_to_tune)
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
            res = res._replace(iteration=iteration,
                               params={k: v.detach().clone() if torch.is_tensor(v) else v for k, v in
                                       self.params.items()})
            logger.info(f"i:{iteration} cost: {res.costs.mean().item()} params:{self.params}")
            self.results.append(res)
        return res

    def define_parameters(self, params_to_tune: typing.Sequence[str]):
        pm = {}
        # take on the assigned values to the MPPI
        if 'sigma' in params_to_tune:
            # we're going to require that sigma be diagonal of positive values to enforce positive definiteness
            pm['sigma'] = torch.cat([self.mppi.noise_sigma[i][i].view(1) for i in range(self.mppi.nu)])
        if 'mu' in params_to_tune:
            pm['mu'] = self.mppi.noise_mu.clone()
        if 'lambda' in params_to_tune:
            pm['lambda'] = self.mppi.lambda_
        if 'horizon' in params_to_tune:
            pm['horizon'] = self.mppi.T
        self.params = pm

    def flatten_params(self):
        x = []
        if 'sigma' in self.params:
            x.append(self.params['sigma'].detach().cpu().numpy())
        if 'mu' in self.params:
            x.append(self.params['mu'].detach().cpu().numpy())
        if 'lambda' in self.params:
            x.append([self.params['lambda']])
        if 'horizon' in self.params:
            x.append([self.params['horizon']])
        x = np.concatenate(x)
        return x

    def unflatten_params(self, x):
        # have to be in the same order as the flattening
        params = {}
        nu = self.mppi.nu
        i = 0
        if 'sigma' in self.params:
            sigma = ensure_tensor(self.d, self.dtype, x[i:i + nu])
            sigma[sigma < self.eps] = self.eps
            params['sigma'] = sigma
            i += nu
        if 'mu' in self.params:
            mu = ensure_tensor(self.d, self.dtype, x[i:i + nu])
            params['mu'] = mu
            i += nu
        if 'lambda' in self.params:
            v = max(x[i], self.eps)
            params['lambda'] = v
            i += 1
        if 'horizon' in self.params:
            v = max(round(x[i]), 1)
            params['horizon'] = v
            i += 1
        self.apply_parameters(params)

    def apply_parameters(self, params):
        if 'sigma' in params:
            # to remain positive definite
            self.mppi.noise_dist = MultivariateNormal(self.mppi.noise_mu, covariance_matrix=torch.diag(params['sigma']))
            self.mppi.noise_sigma_inv = torch.inverse(self.mppi.noise_sigma.detach())
        if 'mu' in params:
            # to remain positive definite
            self.mppi.noise_dist = MultivariateNormal(params['mu'], covariance_matrix=self.mppi.noise_sigma)
            self.mppi.noise_sigma_inv = torch.inverse(self.mppi.noise_sigma.detach())
        if 'lambda' in params:
            self.mppi.lambda_ = params['lambda']
        if 'horizon' in params:
            self.mppi.change_horizon(params['horizon'])
        self.params = params
