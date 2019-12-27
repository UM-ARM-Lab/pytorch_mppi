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

    import random

    randseed = 5
    if randseed is None:
        randseed = random.randint(0, 1000000)
    random.seed(randseed)
    np.random.seed(randseed)
    torch.manual_seed(randseed)
    logger.info("random seed %d", randseed)

    # new hyperparmaeters for approximate dynamics
    H_UNITS = 16
    TRAIN_EPOCH = 150
    BOOT_STRAP_ITER = 100

    nx = 2
    nu = 1
    # network output is state residual
    network = torch.nn.Sequential(
        torch.nn.Linear(nx + nu, H_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(H_UNITS, H_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(H_UNITS, nx)
    ).double()


    def dynamics(state, perturbed_action):
        u = torch.clamp(perturbed_action, ACTION_LOW, ACTION_HIGH)
        if state.dim() is 1 or u.dim() is 1:
            state = state.view(1, -1)
            u = u.view(1, -1)
        if u.shape[1] > 1:
            u = u[:, 0].view(-1, 1)
        xu = torch.cat((state, u), dim=1)
        state_residual = network(xu)
        next_state = state + state_residual
        next_state[:, 0] = angle_normalize(next_state[:, 0])
        return next_state


    def angular_diff_batch(a, b):
        """Angle difference from b to a (a - b)"""
        d = a - b
        d[d > math.pi] -= 2 * math.pi
        d[d < -math.pi] += 2 * math.pi
        return d


    def angle_normalize(x):
        return (((x + math.pi) % (2 * math.pi)) - math.pi)


    def running_cost(state, action):
        theta = state[:, 0]
        theta_dt = state[:, 1]
        action = action[:, 0]
        cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt ** 2 + 0.001 * action ** 2
        return cost


    dataset = None


    def train(new_data):
        global dataset
        # not normalized inside the simulator
        new_data[:, 0] = angle_normalize(new_data[:, 0])
        if not torch.is_tensor(new_data):
            new_data = torch.from_numpy(new_data)
        # clamp actions
        new_data[:, -1] = torch.clamp(new_data[:, -1], ACTION_LOW, ACTION_HIGH)
        # append data to whole dataset
        if dataset is None:
            dataset = new_data
        else:
            dataset = torch.cat((dataset, new_data), dim=0)

        # train on the whole dataset (assume small enough we can train on all together)
        XU = dataset
        dtheta = angular_diff_batch(XU[1:, 0], XU[:-1, 0])
        dtheta_dt = XU[1:, 1] - XU[:-1, 1]
        Y = torch.cat((dtheta.view(-1, 1), dtheta_dt.view(-1, 1)), dim=1)  # x' - x residual
        XU = XU[:-1]  # make same size as Y

        # thaw network
        for param in network.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(network.parameters())
        for epoch in range(TRAIN_EPOCH):
            optimizer.zero_grad()
            # MSE loss
            Yhat = network(XU)
            loss = (Y - Yhat).norm(2, dim=1) ** 2
            loss.mean().backward()
            optimizer.step()
            logger.info("ds %d epoch %d loss %f", dataset.shape[0], epoch, loss.mean().item())

        # freeze network
        for param in network.parameters():
            param.requires_grad = False


    downward_start = True
    env = gym.make(ENV_NAME).env  # bypass the default TimeLimit wrapper
    env.reset()
    if downward_start:
        env.state = [np.pi, 1]

    # bootstrap network with random actions
    if BOOT_STRAP_ITER:
        logger.info("bootstrapping with random action for %d actions", BOOT_STRAP_ITER)
        new_data = np.zeros((BOOT_STRAP_ITER, nx + nu))
        for i in range(BOOT_STRAP_ITER):
            pre_action_state = env.state
            action = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH)
            env.step([action])
            # env.render()
            new_data[i, :nx] = pre_action_state
            new_data[i, nx:] = action

        train(new_data)
        logger.info("bootstrapping finished")

    env = wrappers.Monitor(env, '/tmp/mppi/', force=True)
    env.reset()
    if downward_start:
        env.env.state = [np.pi, 1]

    nx = 2
    mppi_gym = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                         lambda_=lambda_)
    mppi.run_mppi(mppi_gym, env, train)
