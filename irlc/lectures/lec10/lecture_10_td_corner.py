# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.lec09.lecture_10_mc_q_estimation import keyboard_play
from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment
from irlc.ex10.td0_evaluate import TD0ValueAgent

if __name__ == "__main__":
    env = SuttonCornerGridEnvironment(render_mode='human')
    agent = TD0ValueAgent(env, gamma=1, alpha=0.5)
    keyboard_play(env,agent,method_label='TD(0) (alpha=0.5)')
