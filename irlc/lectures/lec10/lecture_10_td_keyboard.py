# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.lec09.lecture_10_mc_q_estimation import automatic_play_value
from irlc.gridworld.gridworld_environments import BookGridEnvironment
from irlc.ex10.td0_evaluate import TD0ValueAgent
from irlc.lectures.lec10.utils import agent_reset

if __name__ == "__main__":
    env = BookGridEnvironment(render_mode='human', living_reward=-0.05)
    TD0ValueAgent.reset = agent_reset
    agent = TD0ValueAgent(env, gamma=1.0, alpha=0.2)
    automatic_play_value(env,agent,method_label='TD(0)')
