# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc import train
from irlc.ex04.pid_cartpole import PIDCartpoleAgent, get_offbalance_cart

if __name__ == "__main__":
    """
    Second task: We will now also try to bring the cart towards x=0.
    """
    env = get_offbalance_cart(30)
    agent = PIDCartpoleAgent(env, env.dt, ...)
    # TODO: 1 lines missing.
    raise NotImplementedError("Define your agent here (including parameters)")
    _, trajectories = train(env, agent, num_episodes=1, reset=False)  # Note reset=False to maintain initial conditions.
    env.close()
