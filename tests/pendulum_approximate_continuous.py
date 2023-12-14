"""
Same as approximate dynamics, but now the input is sine and cosine of theta (output is still dtheta)
This is a continuous representation of theta, which some papers show is easier for a NN to learn.
"""
import gym
import numpy as np
import torch
import logging
import math
from pytorch_mppi import mppi
from gym import logger as gym_log

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

if __name__ == "__main__":
    ENV_NAME = "Pendulum-v1"
    TIMESTEPS = 15  # T
    N_SAMPLES = 100  # K
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0

    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double

    noise_sigma = torch.tensor(1, device=d, dtype=dtype)
    # noise_sigma = torch.tensor([[10, 0], [0, 10]], device=d, dtype=dtype)
    lambda_ = 1.

    import random

    randseed = 24
    if randseed is None:
        randseed = random.randint(0, 1000000)
    random.seed(randseed)
    np.random.seed(randseed)
    torch.manual_seed(randseed)
    logger.info("random seed %d", randseed)

    # new hyperparmaeters for approximate dynamics
    H_UNITS = 32
    TRAIN_EPOCH = 150
    BOOT_STRAP_ITER = 100

    nx = 2
    nu = 1
    # network output is state residual
    network = torch.nn.Sequential(
        torch.nn.Linear(nx + nu + 1, H_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(H_UNITS, H_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(H_UNITS, nx)
    ).double().to(device=d)


    def dynamics(state, perturbed_action):
        u = torch.clamp(perturbed_action, ACTION_LOW, ACTION_HIGH)
        if state.dim() == 1 or u.dim() == 1:
            state = state.view(1, -1)
            u = u.view(1, -1)
        if u.shape[1] > 1:
            u = u[:, 0].view(-1, 1)
        xu = torch.cat((state, u), dim=1)
        # feed in cosine and sine of angle instead of theta
        xu = torch.cat((torch.sin(xu[:, 0]).view(-1, 1), torch.cos(xu[:, 0]).view(-1, 1), xu[:, 1:]), dim=1)
        state_residual = network(xu)
        # output dtheta directly so can just add
        next_state = state + state_residual
        next_state[:, 0] = angle_normalize(next_state[:, 0])
        return next_state


    def true_dynamics(state, perturbed_action):
        # true dynamics from gym
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)

        g = 10
        m = 1
        l = 1
        dt = 0.05

        u = perturbed_action
        u = torch.clamp(u, -2, 2)

        newthdot = thdot + (3 * g / (2 * l) * torch.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = torch.clip(newthdot, -8, 8)
        newth = th + newthdot * dt

        state = torch.cat((newth, newthdot), dim=1)
        return state


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
        cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt ** 2
        return cost


    dataset = None
    # create some true dynamics validation set to compare model against
    Nv = 1000
    statev = torch.cat(((torch.rand(Nv, 1, dtype=torch.double, device=d) - 0.5) * 2 * math.pi,
                        (torch.rand(Nv, 1, dtype=torch.double, device=d) - 0.5) * 16), dim=1)
    actionv = (torch.rand(Nv, 1, dtype=torch.double, device=d) - 0.5) * (ACTION_HIGH - ACTION_LOW)


    def train(new_data):
        global dataset
        # not normalized inside the simulator
        new_data[:, 0] = angle_normalize(new_data[:, 0])
        if not torch.is_tensor(new_data):
            new_data = torch.from_numpy(new_data)
        # clamp actions
        new_data[:, -1] = torch.clamp(new_data[:, -1], ACTION_LOW, ACTION_HIGH)
        new_data = new_data.to(device=d)
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
        xu = XU[:-1]  # make same size as Y
        xu = torch.cat((torch.sin(xu[:, 0]).view(-1, 1), torch.cos(xu[:, 0]).view(-1, 1), xu[:, 1:]), dim=1)

        # thaw network
        for param in network.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(network.parameters())
        for epoch in range(TRAIN_EPOCH):
            optimizer.zero_grad()
            # MSE loss
            Yhat = network(xu)
            loss = (Y - Yhat).norm(2, dim=1) ** 2
            loss.mean().backward()
            optimizer.step()
            logger.debug("ds %d epoch %d loss %f", dataset.shape[0], epoch, loss.mean().item())

        # freeze network
        for param in network.parameters():
            param.requires_grad = False

        # evaluate network against true dynamics
        yt = true_dynamics(statev, actionv)
        yp = dynamics(statev, actionv)
        dtheta = angular_diff_batch(yp[:, 0], yt[:, 0])
        dtheta_dt = yp[:, 1] - yt[:, 1]
        E = torch.cat((dtheta.view(-1, 1), dtheta_dt.view(-1, 1)), dim=1).norm(dim=1)
        logger.info("Error with true dynamics theta %f theta_dt %f norm %f", dtheta.abs().mean(),
                    dtheta_dt.abs().mean(), E.mean())
        logger.debug("Start next collection sequence")


    downward_start = True
    env = gym.make(ENV_NAME, render_mode="human")  # bypass the default TimeLimit wrapper
    env.reset()
    if downward_start:
        env.state = env.unwrapped.state = [np.pi, 1]

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

    env.reset()
    if downward_start:
        env.state = env.unwrapped.state = [np.pi, 1]

    mppi_gym = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                         lambda_=lambda_, device=d, u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
                         u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))
    total_reward, data = mppi.run_mppi(mppi_gym, env, train)
    logger.info("Total reward %f", total_reward)
