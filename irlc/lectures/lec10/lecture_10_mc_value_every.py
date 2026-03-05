# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.exam_tabular_examples.helper import keyboard_play_value
# from irlc.berkley.rl.feature_encoder import SimplePacmanExtractor
from irlc.gridworld.gridworld_environments import BookGridEnvironment
from irlc.ex09.mc_evaluate import MCEvaluationAgent

if __name__ == "__main__":
    env = BookGridEnvironment(view_mode=1, render_mode='human', living_reward=-0.05)
    agent = MCEvaluationAgent(env, gamma=1, alpha=None, first_visit=False)

    keyboard_play_value(env,agent,method_label='MC every')
