import numpy as np

# pip install ribs
import ribs

from pytorch_mppi import autotune
from pytorch_mppi.autotune_global import AutotuneGlobal


class CMAMEOpt(autotune.Optimizer):
    """Quality Diversity optimize using CMA-ME to find a set of good and diverse hyperparameters"""

    def __init__(self, population=10, sigma=1.0, bins=15):
        """

        :param population: number of parameters to sample at once (scales linearly)
        :param sigma: initial variance along all dimensions
        :param bins: int or a Sequence[int] for each hyperparameter for the number of bins in the archive.
        More bins means more granularity along that dimension.
        """
        self.population = population
        self.sigma = sigma
        self.archive = None
        self.qd_score_offset = -3000
        self.num_emitters = 1
        self.bins = bins
        super().__init__()

    def setup_optimization(self):
        if not isinstance(self.tuner, AutotuneGlobal):
            raise RuntimeError(f"Quality diversity optimizers require global search space information provided "
                               f"by AutotuneMPPIGlobal")

        x = self.tuner.flatten_params()
        ranges = self.tuner.linearized_search_space()
        ranges = list(ranges.values())

        param_dim = len(x)
        bins = self.bins
        if isinstance(bins, (int, float)):
            bins = [bins for _ in range(param_dim)]
        self.archive = ribs.archives.GridArchive(solution_dim=param_dim,
                                                 dims=bins,
                                                 ranges=ranges,
                                                 seed=np.random.randint(0, 10000), qd_score_offset=self.qd_score_offset)
        emitters = [
            ribs.emitters.EvolutionStrategyEmitter(self.archive, x0=x, sigma0=self.sigma, batch_size=self.population,
                                                   seed=np.random.randint(0, 10000)) for i in
            range(self.num_emitters)
        ]
        self.optim = ribs.schedulers.Scheduler(self.archive, emitters)

    def optimize_step(self):
        if not isinstance(self.tuner, AutotuneGlobal):
            raise RuntimeError(f"Quality diversity optimizers require global search space information provided "
                               f"by AutotuneMPPIGlobal")

        params = self.optim.ask()
        # measure is the whole hyperparameter set - we want to diverse along each dimension

        cost_per_param = []
        all_rollouts = []
        bcs = []
        for param in params:
            full_param = self.tuner.unflatten_params(param)
            res = self.tuner.evaluate_fn()
            cost_per_param.append(res.costs.mean().cpu().numpy())
            all_rollouts.append(res.rollouts)
            behavior = self.tuner.linearize_params(full_param)
            bcs.append(behavior)

        cost_per_param = np.array(cost_per_param)
        self.optim.tell(-cost_per_param, bcs)

        best_param = self.archive.best_elite
        # best_param = self.optim.best.x
        self.tuner.unflatten_params(best_param.solution)
        res = self.tuner.evaluate_fn()
        return res

    def get_diverse_top_parameters(self, num_top):
        df = self.archive.as_pandas()
        objectives = df.objective_batch()
        solutions = df.solution_batch()
        # store to allow restoring on next step
        if len(solutions) > num_top:
            order = np.argpartition(-objectives, num_top)
            solutions = solutions[order[:num_top]]

        return [self.tuner.unflatten_params(x, apply=False) for x in solutions]
