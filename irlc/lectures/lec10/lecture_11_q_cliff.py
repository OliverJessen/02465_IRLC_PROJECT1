# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.gridworld.gridworld_environments import CliffGridEnvironment2
from irlc.ex10.q_agent import QAgent
from irlc.lectures.lec10.lecture_11_sarsa_cliff import cliffwalk, gamma, alpha, epsi

if __name__ == "__main__":
    import numpy as np
    np.random.seed(1)
    env = CliffGridEnvironment2(zoom=.8, render_mode='human')
    agent = QAgent(env, gamma=gamma, epsilon=epsi, alpha=alpha)
    cliffwalk(env, agent, method_label="Q-learning")
