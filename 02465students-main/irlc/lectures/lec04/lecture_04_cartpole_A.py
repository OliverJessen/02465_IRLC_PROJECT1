# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc import train
from irlc.ex04.pid_cartpole import PIDCartpoleAgent, get_offbalance_cart

if __name__ == "__main__":
    env = get_offbalance_cart(30)
    agent = PIDCartpoleAgent(env, dt=env.dt, Kp=120, Ki=0, Kd=10, balance_to_x0=False)
    _, trajectories = train(env, agent, num_episodes=1, reset=False)
    env.close()
