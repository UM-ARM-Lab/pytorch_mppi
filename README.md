# PyTorch MPPI Implementation
This repository implements Model Predictive Path Integral (MPPI) 
with approximate dynamics in pytorch. MPPI typically requires actual
trajectory samples, but [this paper](https://ieeexplore.ieee.org/document/7989202/)
showed that it could be done with approximate dynamics (such as with a neural network)
using importance sampling.

Thus it can be used in place of other trajectory optimization methods
such as the Cross Entropy Method (CEM), or random shooting.

# Usage
Clone repository somewhere, then `pip3 install -e .` to install in editable mode.
See `tests/pendulum_approximate.py` for usage with a neural network approximating
the pendulum dynamics.

# Features
- Approximate dynamics MPPI with importance sampling
- Parallel/batch pytorch implementation for accelerated sampling

# Tests
Under `tests` you can find the `MPPI` method applied to known pendulum dynamics
and approximate pendulum dynamics (with a 2 layer feedforward net 
estimating the state residual).

Sample result on approximate dynamics with 100 steps of random policy data
to initialize the dynamics:

![pendulum results](https://i.imgur.com/euYQJ25.gif)