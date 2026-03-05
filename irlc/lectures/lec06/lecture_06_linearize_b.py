# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc import plot_trajectory, train
from irlc.ex06.linearization_agent import get_offbalance_cart, LinearizationAgent
import numpy as np
import matplotlib
# matplotlib.use("tkagg")
import matplotlib.pyplot as plt


if __name__ == "__main__":
    np.random.seed(42) # I don't think these results are seed-dependent but let's make sure.
    env = get_offbalance_cart(4, sleep_time=0.08) # Simulate for a little time to get an off-balance cart. Increase 4-->10 to get failure.
    agent = LinearizationAgent(env, model=env.discrete_model, xbar=env.discrete_model.x_upright, ubar=env.action_space.sample()*0)
    _, trajectories = train(env, agent, num_episodes=1, return_trajectory=True, reset=False)  # Note reset=False to maintain initial conditions.
    plt.figure()
    plot_trajectory(trajectories[0], env, xkeys=[0, 2, 3], ukeys=[0])
    plt.show()
    env.close()
