# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.lec10.lecture_10_mc_q_estimation import keyboard_play
from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment, BookGridEnvironment
from irlc.lectures.lec10.lecture_10_mc_value_first_one_state import MCAgentOneState
from irlc.ex09.mc_agent import MCAgent
from irlc.ex09.mc_evaluate import MCEvaluationAgent
import numpy as np
from irlc import interactive, train


if __name__ == "__main__":
    env = BookGridEnvironment(render_mode='human', living_reward=-0.05)
    agent = MCAgentOneState(env, gamma=1, alpha=None, first_visit=True, state=(0,2))
    method_label = 'MC (gamma=1)'
    agent.label = method_label
    autoplay = False
    env, agent = interactive(env, agent, autoplay=autoplay)
    # agent = PlayWrapper(agent, env,autoplay=autoplay)
    # env = VideoMonitor(env, agent=agent, fps=100, agent_monitor_keys=('pi', 'Q'), render_kwargs={'method_label': method_label})
    num_episodes = 1000
    train(env, agent, num_episodes=num_episodes)
    env.close()

    # keyboard_play(env,agent,method_label='MC (alpha=0.5)')
