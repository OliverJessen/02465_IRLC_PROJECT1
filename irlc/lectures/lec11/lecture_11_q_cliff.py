# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.gridworld.gridworld_environments import CliffGridEnvironment2
from irlc.ex10.q_agent import QAgent


# def cliffwalk(env, agent, method_label="method"):
#     agent = PlayWrapper(agent, env)
    # env = VideoMonitor(env, agent=agent, fps=100, continious_recording=True, agent_monitor_keys=('pi', 'Q'), render_kwargs={'method_label': method_label})
    # train(env, agent, num_episodes=200)
    # env.close()

from irlc.lectures.lec11.lecture_11_sarsa_cliff import cliffwalk, gamma, alpha, epsi
if __name__ == "__main__":
    import numpy as np
    np.random.seed(1)
    env = CliffGridEnvironment2(zoom=.8, render_mode='human')
    agent = QAgent(env, gamma=gamma, epsilon=epsi, alpha=alpha)
    cliffwalk(env, agent, method_label="Q-learning")
