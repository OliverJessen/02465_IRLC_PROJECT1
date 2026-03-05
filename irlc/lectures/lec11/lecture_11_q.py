# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
# from irlc.berkley.rl.feature_encoder import SimplePacmanExtractor
from irlc.gridworld.gridworld_environments import BookGridEnvironment
from irlc.lectures.lec10.lecture_10_mc_q_estimation import keyboard_play
from irlc.ex10.q_agent import QAgent

if __name__ == "__main__":
    env = BookGridEnvironment(render_mode='human')
    agent = QAgent(env, gamma=0.95, epsilon=0.1, alpha=.2)
    keyboard_play(env, agent, method_label="Q-learning")
