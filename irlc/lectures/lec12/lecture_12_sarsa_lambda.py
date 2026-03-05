# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.lec12.sarsa_lambda_delay import SarsaLambdaDelayAgent

if __name__ == "__main__":
    from irlc.gridworld.gridworld_environments import BookGridEnvironment
    from irlc.lectures.lec10.lecture_10_mc_q_estimation import keyboard_play

    frames_per_second = 30

    env = BookGridEnvironment(render_mode='human', frames_per_second=frames_per_second, living_reward=-0.05)
    agent = SarsaLambdaDelayAgent(env, gamma=1., epsilon=0, alpha=0.1)
    method_label = f"SarsaLambda (gamma=0.99, epsilon=0.1, alpha=0.5)"
    keyboard_play(env, agent, method_label=method_label)

    #
    #
    # open_play(SarsaLambdaDelayAgent, method_label="Sarsa(Lambda)", lamb=0.9)
