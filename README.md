# PyTorch MPPI Implementation
This repository implements Model Predictive Path Integral (MPPI) 
with approximate dynamics in pytorch. MPPI typically requires actual
trajectory samples, but [this paper](https://ieeexplore.ieee.org/document/7989202/)
showed that it could be done with approximate dynamics (such as with a neural network)
using importance sampling.

Thus it can be used in place of other trajectory optimization methods
such as the Cross Entropy Method (CEM), or random shooting.

# Installation
```shell
pip install pytorch-mppi
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

# Tests
You'll need to install `gym<=0.20` to run the tests (for the `Pendulum-v0` environment).
The easy way to install this and other testing dependencies is running `python setup.py test`.
Note that `gym` past `0.20` deprecated `Pendulum-v0` for `Pendulum-v1` with incompatible dynamics.

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
