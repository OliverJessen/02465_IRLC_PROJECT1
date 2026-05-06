# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.gridworld.gridworld_environments import OpenGridEnvironment
from irlc.lectures.lec09.lecture_10_mc_q_estimation import keyboard_play
from irlc.ex10.sarsa_agent import SarsaAgent

def open_play(Agent, method_label, frames_per_second=30, **args):
    env = OpenGridEnvironment(render_mode='human', frames_per_second=frames_per_second)
    agent = Agent(env, gamma=0.99, epsilon=0.1, alpha=.5, **args)
    method_label = f"{method_label} (gamma=0.99, epsilon=0.1, alpha=0.5)"
    keyboard_play(env, agent, method_label=method_label)

if __name__ == "__main__":
    open_play(SarsaAgent, method_label="Sarsa")
