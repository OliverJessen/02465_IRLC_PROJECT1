# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.utils.bandit_graphics_environment import GraphicalBandit
from irlc import interactive, train
# import numpy as np
import time

def bandit_ucb(autoplay=False):
    env = GraphicalBandit(10, render_mode='human', frames_per_second=30)
    env.reset()
    #env.viewer.show_q_star = True
    #env.viewer.show_q_ucb = True
    from irlc.ex07.ucb_agent import UCBAgent
    agent = UCBAgent(env, c=1)
    agent.method = 'UCB'

    env, agent = interactive(env, agent, autoplay=autoplay)
    t0 = time.time()
    n = 5000
    stats, _ = train(env, agent, max_steps=n, num_episodes=10, return_trajectory=False, verbose=False)
    tpf = (time.time() - t0) / n
    print("tpf", tpf, 'fps', 1 / tpf)
    env.close()


if __name__ == "__main__":
    bandit_ucb()
