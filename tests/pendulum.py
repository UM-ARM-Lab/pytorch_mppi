import gym
import numpy as np
import torch
import logging
import math
from pytorch_mppi import mppi
from gym import wrappers, logger as gym_log

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

if __name__ == "__main__":
    ENV_NAME = "Pendulum-v0"
    TIMESTEPS = 15  # T
    N_SAMPLES = 100  # K
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0

    d = "cpu"
    dtype = torch.double

    noise_sigma = torch.tensor(10, device=d, dtype=dtype)
    # noise_sigma = torch.tensor([[10, 0], [0, 10]], device=d, dtype=dtype)
    lambda_ = 1.


    def dynamics(state, perturbed_action):
        # true dynamics from gym
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)

        g = 10
        m = 1
        l = 1
        dt = 0.05

        u = perturbed_action
        u = torch.clamp(u, -2, 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -8, 8)

        state = torch.cat((newth, newthdot), dim=1)
        return state


    def angle_normalize(x):
        return (((x + math.pi) % (2 * math.pi)) - math.pi)


    def running_cost(state, action):
        theta = state[:, 0]
        theta_dt = state[:, 1]
        action = action[:, 0]
        cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt ** 2
        return cost


    def train(new_data):
        pass


    downward_start = True
    env = gym.make(ENV_NAME).env  # bypass the default TimeLimit wrapper
    env.reset()
    if downward_start:
        env.state = [np.pi, 1]

    env = wrappers.Monitor(env, '/tmp/mppi/', force=True)
    env.reset()
    if downward_start:
        env.env.state = [np.pi, 1]

    nx = 2
    mppi_gym = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                         lambda_=lambda_)
    total_reward = mppi.run_mppi(mppi_gym, env, train)
    logger.info("Total reward %f", total_reward)
