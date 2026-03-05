# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.lec10.lecture_10_mc_q_estimation import keyboard_play
from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment
from irlc.ex09.mc_agent import MCAgent
import numpy as np

if __name__ == "__main__":
    env = SuttonCornerGridEnvironment(render_mode='human')
    agent = MCAgent(env, gamma=1, epsilon=1, alpha=.5, first_visit=False)
    keyboard_play(env,agent,method_label='MC (alpha=0.5)')
