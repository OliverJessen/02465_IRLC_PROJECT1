# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
# from irlc.utils.player_wrapper_pyglet import PlayWrapper
from irlc.gridworld.gridworld_environments import CliffGridEnvironment2
# from irlc.utils.video_monitor import VideoMonitor
from irlc.ex01.agent import train
from irlc import interactive
from irlc.ex10.sarsa_agent import SarsaAgent


def cliffwalk(env, agent, method_label="method"):
    # agent = PlayWrapper(agent, env)
    env.label = method_label
    agent.method_label = method_label
    agent.label = method_label
    agent.method = method_label


    env, agent = interactive(env, agent)
    # env = VideoMonitor(env, agent=agent, fps=200, continious_recording=True, agent_monitor_keys=('pi', 'Q'), render_kwargs={'method_label': method_label})
    train(env, agent, num_episodes=1000)
    env.close()

epsi = 0.5
gamma = 1.0
alpha = .3

if __name__ == "__main__":
    import numpy as np
    np.random.seed(1)
    env = CliffGridEnvironment2(zoom=.8, render_mode='human')
    agent = SarsaAgent(env, gamma=gamma, epsilon=epsi, alpha=alpha)
    # agent = QAgent(env, gamma=0.95, epsilon=0.5, alpha=.2)
    cliffwalk(env, agent, method_label="Sarsa")
