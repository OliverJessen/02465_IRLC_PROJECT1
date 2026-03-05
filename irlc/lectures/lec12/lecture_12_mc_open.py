# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
# from irlc.berkley.rl.feature_encoder import SimplePacmanExtractor
# from irlc.lectures.lecture_09_mc import keyboard_play

# alpha = 0.5
# gamma =

# def open_play(Agent, method_label, **args):
#     env = OpenGridEnvironment()
#     agent = Agent(env, gamma=0.95, epsilon=0.1, alpha=.5, **args)
#     keyboard_play(env, agent, method_label=method_label)

from irlc.lectures.lec11.lecture_10_sarsa_open import open_play
from irlc.ex09.mc_agent import MCAgent
if __name__ == "__main__":
    # env = OpenGridEnvironment()
    # agent = (env, gamma=0.95, epsilon=0.1, alpha=.5)
    open_play(MCAgent, method_label="MC agent")
    #
