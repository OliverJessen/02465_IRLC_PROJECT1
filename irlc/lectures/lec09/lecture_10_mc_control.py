# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.lec09.lecture_10_mc_q_estimation import keyboard_play
from irlc.gridworld.gridworld_environments import BookGridEnvironment
from irlc.ex09.mc_agent import MCAgent
# from irlc.lectures.lec10.utils import MCAgentResettable

import numpy as np

if __name__ == "__main__":
    np.random.seed(433)
    env = BookGridEnvironment(render_mode='human',zoom=2, living_reward=-0.05)
    # agent = MCAgent(env, gamma=0.9, epsilon=0.15, alpha=0.1, first_visit=True)
    from irlc.lectures.lec10.utils import agent_reset
    MCAgent.reset = agent_reset
    agent = MCAgent(env, gamma=1.0, epsilon=0.15, alpha=None, first_visit=True)

    # env, agent = interactive(env, agent)
    keyboard_play(env,agent,method_label='MC control')
