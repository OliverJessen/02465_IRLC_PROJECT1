# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.gridworld.gridworld_environments import BookGridEnvironment
from irlc.lectures.lec09.lecture_10_mc_q_estimation import keyboard_play
from irlc.lectures.lec10.sarsa_nstep_delay import SarsaDelayNAgent

if __name__ == "__main__":
    env = BookGridEnvironment(render_mode='human')
    agent = SarsaDelayNAgent(env, gamma=0.95, epsilon=0.1, alpha=.96, n=1)
    keyboard_play(env, agent, method_label="Sarsa")
