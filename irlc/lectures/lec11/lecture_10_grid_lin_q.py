# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.berkley.rl.semi_grad_q import LinearSemiGradQAgent
from irlc.ex11.feature_encoder import GridworldXYEncoder
from irlc.gridworld.gridworld_environments import BookGridEnvironment
from irlc.lectures.lec10.lecture_10_mc_q_estimation import keyboard_play

if __name__ == "__main__":
    env = BookGridEnvironment(render_mode='human')
    agent = LinearSemiGradQAgent(env, gamma=0.95, epsilon=0.1, alpha=.01, q_encoder=GridworldXYEncoder(env))
    keyboard_play(env, agent, method_label="Q-lin-xy")
