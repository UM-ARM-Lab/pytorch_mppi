# PyTorch MPPI Implementation
This repository implements Model Predictive Path Integral (MPPI) 
with approximate dynamics in pytorch. MPPI typically requires actual
trajectory samples, but [this paper](https://ieeexplore.ieee.org/document/7989202/)
showed that it could be done with approximate dynamics (such as with a neural network)
using importance sampling.

Thus it can be used in place of other trajectory optimization methods
such as the Cross Entropy Method (CEM), or random shooting.

---
New since Aug 2024 smoothing methods, including our own KMPPI, see the section below on smoothing

# Installation
```shell
pip install pytorch-mppi
```
for autotuning hyperparameters, install with
```shell
pip install pytorch-mppi[tune]
```

for running tests, install with
```shell
pip install pytorch-mppi[test]
```
for development, clone the repository then install in editable mode
```shell
pip install -e .
```

# Usage
See `tests/pendulum_approximate.py` for usage with a neural network approximating
the pendulum dynamics. See the `not_batch` branch for an easier to read
algorithm. Basic use case is shown below

```python
from pytorch_mppi import MPPI

# create controller with chosen parameters
ctrl = MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
            lambda_=lambda_, device=d,
            u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
            u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))

# assuming you have a gym-like env
obs = env.reset()
for i in range(100):
    action = ctrl.command(obs)
    obs, reward, done, _ = env.step(action.cpu().numpy())
```

# Requirements
- pytorch (>= 1.0)
- `next state <- dynamics(state, action)` function (doesn't have to be true dynamics)
    - `state` is `K x nx`, `action` is `K x nu`
- `cost <- running_cost(state, action)` function
    - `cost` is `K x 1`, state is `K x nx`, `action` is `K x nu`

# Features
- Approximate dynamics MPPI with importance sampling
- Parallel/batch pytorch implementation for accelerated sampling
- Control bounds via sampling control noise from rectified gaussian 
- Handle stochastic dynamic models (assuming each call is a sample) by sampling multiple state trajectories for the same
action trajectory with `rollout_samples`
- 
# Parameter tuning and hints
`terminal_state_cost` - function(state (K x T x nx)) -> cost (K x 1) by default there is no terminal
cost, but if you experience your trajectory getting close to but never quite reaching the goal, then
having a terminal cost can help. The function should scale with the horizon (T) to keep up with the
scaling of the running cost.

`lambda_` - higher values increases the cost of control noise, so you end up with more
samples around the mean; generally lower values work better (try `1e-2`)

`num_samples` - number of trajectories to sample; generally the more the better.
Runtime performance scales much better with `num_samples` than `horizon`, especially
if you're using a GPU device (remember to pass that in!)

`noise_mu` - the default is 0 for all control dimensions, which may work out
really poorly if you have control bounds and the allowed range is not 0-centered.
Remember to change this to an appropriate value for non-symmetric control dimensions.

## Smoothing
From version 0.8.0 onwards, you can use MPPI variants that smooth the control signal. We've implemented
[SMPPI](https://arxiv.org/pdf/2112.09988) as well our own kernel interpolation MPPI (KMPPI). In the base algorithm,
you can achieve somewhat smoother trajectories by increasing `lambda_`; however, that comes at the cost of
optimality. Explicit smoothing algorithms can achieve smoothness without sacrificing optimality.

We used it and described it in our recent paper ([arxiv](https://arxiv.org/abs/2408.10450)) and you can cite it 
until we release a work dedicated to KMPPI. Below we show the difference between MPPI, SMPPI, and KMPPI on a toy
2D navigation problem where the control is a constrained delta position. You can check it out in `tests/smooth_mppi.py`.

The API is mostly the same, with some additional constructor options:
```python
ctrl = mppi.KMPPI(args, 
                 kernel=mppi.RBFKernel(sigma=2), # kernel in trajectory time space (1 dimensional)
                 num_support_pts=5,              # number of control points to sample, <= horizon
                 **kwargs)
```
The kernel can be any subclass of `mppi.TimeKernel`. It is a kernel in the trajectory time space (1 dimensional).
Note that B-spline smoothing can be achieved by using a B-spline kernel. The number of support points is the number
of control points to sample. Any trajectory points in between are interpolated using the kernel. For example if a
trajectory horizon is 20 and `num_support_pts` is 5, then 5 control points evenly spaced throughout the horizon
(with the first and last corresponding to the actual start and end of the trajectory) are sampled. The rest of the
trajectory is interpolated using the kernel. The kernel is applied to the control signal, not the state signal.

MPPI without smoothing
![MPPI](https://imgur.com/aXSo3Ib.gif) 
[SMPPI](https://arxiv.org/pdf/2112.09988) smoothing by sampling noise in the action derivative space doesn't work well on this problem
![SMPPI](https://imgur.com/y1hvqlD.gif)
KMPPI smoothing with RBF kernel works well
![KMPPI](https://imgur.com/mZmbC4S.gif)



## Autotune
from version 0.5.0 onwards, you can automatically tune the hyperparameters.
A convenient tuner compatible with the popular [ray tune](https://docs.ray.io/en/latest/tune/index.html) library
is implemented. You can select from a variety of cutting edge black-box optimizers such as 
[CMA-ES](https://github.com/CMA-ES/pycma), [HyperOpt](http://hyperopt.github.io/hyperopt/),
[fmfn/BayesianOptimization](https://github.com/fmfn/BayesianOptimization), and so on.
See `tests/auto_tune_parameters.py` for an example. A tutorial based on it follows.

The tuner can be used for other controllers as well, but you will need to define the appropriate
`TunableParameter` subclasses.

First we create a toy 2D environment to do controls on and create the controller with some
default parameters.
```python
import torch
from pytorch_mppi import MPPI

device = "cpu"
dtype = torch.double

# create toy environment to do on control on (default start and goal)
env = Toy2DEnvironment(visualize=True, terminal_scale=10)

# create MPPI with some initial parameters
mppi = MPPI(env.dynamics, env.running_cost, 2,
            terminal_state_cost=env.terminal_cost,
            noise_sigma=torch.diag(torch.tensor([5., 5.], dtype=dtype, device=device)),
            num_samples=500,
            horizon=20, device=device,
            u_max=torch.tensor([2., 2.], dtype=dtype, device=device),
            lambda_=1)
```

We then need to create an evaluation function for the tuner to tune on. 
It should take no arguments and output a `EvaluationResult` populated at least by costs.
If you don't need rollouts for the cost evaluation, then you can set it to None in the return.
Tips for creating the evaluation function are described in comments below:

```python
from pytorch_mppi import autotune
# use the same nominal trajectory to start with for all the evaluations for fairness
nominal_trajectory = mppi.U.clone()
# parameters for our sample evaluation function - lots of choices for the evaluation function
evaluate_running_cost = True
num_refinement_steps = 10
num_trajectories = 5

def evaluate():
    costs = []
    rollouts = []
    # we sample multiple trajectories for the same start to goal problem, but in your case you should consider
    # evaluating over a diverse dataset of trajectories
    for j in range(num_trajectories):
        mppi.U = nominal_trajectory.clone()
        # the nominal trajectory at the start will be different if the horizon's changed
        mppi.change_horizon(mppi.T)
        # usually MPPI will have its nominal trajectory warm-started from the previous iteration
        # for a fair test of tuning we will reset its nominal trajectory to the same random one each time
        # we manually warm it by refining it for some steps
        for k in range(num_refinement_steps):
            mppi.command(env.start, shift_nominal_trajectory=False)

        rollout = mppi.get_rollouts(env.start)

        this_cost = 0
        rollout = rollout[0]
        # here we evaluate on the rollout MPPI cost of the resulting trajectories
        # alternative costs for tuning the parameters are possible, such as just considering terminal cost
        if evaluate_running_cost:
            for t in range(len(rollout) - 1):
                this_cost = this_cost + env.running_cost(rollout[t], mppi.U[t])
        this_cost = this_cost + env.terminal_cost(rollout, mppi.U)

        rollouts.append(rollout)
        costs.append(this_cost)
    # can return None for rollouts if they do not need to be calculated
    return autotune.EvaluationResult(torch.stack(costs), torch.stack(rollouts))
```

With this we have enough to start tuning. For example, we can tune iteratively with the CMA-ES optimizer

```python
# these are subclass of TunableParameter (specifically MPPIParameter) that we want to tune
params_to_tune = [autotune.SigmaParameter(mppi), autotune.HorizonParameter(mppi), autotune.LambdaParameter(mppi)]
# create a tuner with a CMA-ES optimizer
tuner = autotune.Autotune(params_to_tune, evaluate_fn=evaluate, optimizer=autotune.CMAESOpt(sigma=1.0))
# tune parameters for a number of iterations
iterations = 30
for i in range(iterations):
  # results of this optimization step are returned
  res = tuner.optimize_step()
  # we can render the rollouts in the environment
  env.draw_rollouts(res.rollouts)
# get best results and apply it to the controller
# (by default the controller will take on the latest tuned parameter, which may not be best)
res = tuner.get_best_result()
tuner.apply_parameters(res.param_values)
```
This is a local search method that optimizes starting from the initially defined parameters.
For global searching, we use ray tune compatible searching algorithms. Note that you can modify the
search space of each parameter, but default reasonable ones are provided.

```python
# can also use a Ray Tune optimizer, see
# https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#search-algorithms-tune-search
# rather than adapting the current parameters, these optimizers allow you to define a search space for each
# and will search on that space
from pytorch_mppi import autotune_global
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch

# the global version of the parameters define a reasonable search space for each parameter
params_to_tune = [autotune_global.SigmaGlobalParameter(mppi),
                  autotune_global.HorizonGlobalParameter(mppi),
                  autotune_global.LambdaGlobalParameter(mppi)]

# be sure to close any figures before ray tune optimization or they will be duplicated
env.visualize = False
plt.close('all')
tuner = autotune_global.AutotuneGlobal(params_to_tune, evaluate_fn=evaluate,
                                       optimizer=autotune_global.RayOptimizer(HyperOptSearch))
# ray tuners cannot be tuned iteratively, but you can specify how many iterations to tune for
res = tuner.optimize_all(100)
res = tuner.get_best_result()
tuner.apply_parameters(res.params)
```

For example tuning hyperparameters (with CMA-ES) only on the toy problem (the nominal trajectory is reset each time so they are sampling from noise):

![toy tuning](https://i.imgur.com/2qtYMwu.gif)

If you want more than just the best solution found, such as if you want diversity
across hyperparameter values, or if your evaluation function has large uncertainty,
then you can directly query past results by
```python
for res in tuner.optim.all_res:
    # the cost
    print(res.metrics['cost'])
    # extract the parameters
    params = tuner.config_to_params(res.config)
    print(params)
    # apply the parameters to the controller
    tuner.apply_parameters(params)
```

Alternatively you can try Quality Diversity optimization using the 
[CMA-ME optimizer](https://github.com/icaros-usc/pyribs). This optimizer will
try to optimize for high quality parameters while ensuring there is diversity across
them. However, it is very slow and you might be better using a `RayOptimizer` and selecting
for top results while checking for diversity.
To use it, you need to install
```python
pip install ribs
```

You then use it as

```python
import pytorch_mppi.autotune_qd

optim = pytorch_mppi.autotune_qd.CMAMEOpt()
tuner = autotune_global.AutotuneGlobal(params_to_tune, evaluate_fn=evaluate,
                                       optimizer=optim)

iterations = 10
for i in range(iterations):
  # results of this optimization step are returned
  res = tuner.optimize_step()
  # we can render the rollouts in the environment
  best_params = optim.get_diverse_top_parameters(5)
  for res in best_params:
    print(res)
```

# Tests
Under `tests` you can find the `MPPI` method applied to known pendulum dynamics
and approximate pendulum dynamics (with a 2 layer feedforward net 
estimating the state residual). Using a continuous angle representation
(feeding `cos(\theta), sin(\theta)` instead of `\theta` directly) makes
a huge difference. Although both works, the continuous representation
is much more robust to controller parameters and random seed. In addition,
the problem of continuing to spin after over-swinging does not appear.

Sample result on approximate dynamics with 100 steps of random policy data
to initialize the dynamics:

![pendulum results](https://i.imgur.com/euYQJ25.gif)

# Related projects
- [pytorch CEM](https://github.com/LemonPi/pytorch_cem) - an alternative MPC shooting method with similar API as this
project
- [pytorch iCEM](https://github.com/UM-ARM-Lab/pytorch_icem) - alternative sampling based MPC
