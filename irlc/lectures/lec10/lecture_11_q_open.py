# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.gridworld.gridworld_environments import OpenGridEnvironment
from irlc.lectures.lec09.lecture_10_mc_q_estimation import keyboard_play
from irlc.ex10.q_agent import QAgent

def open_play(Agent, method_label, **args):
    env = OpenGridEnvironment(render_mode='human')
    agent = Agent(env, gamma=0.99, epsilon=0.1, alpha=.5, **args)
    keyboard_play(env, agent, method_label=method_label)

if __name__ == "__main__":
    open_play(QAgent, method_label="Q-learning")
