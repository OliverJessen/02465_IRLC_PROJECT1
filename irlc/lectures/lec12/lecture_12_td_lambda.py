# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.lec12.td_lambda import TDLambdaAgent

if __name__ == "__main__":
    from irlc.lectures.lec10.lecture_10_mc_q_estimation import keyboard_play
    from irlc.gridworld.gridworld_environments import OpenGridEnvironment

    env = OpenGridEnvironment(render_mode='human', frames_per_second=30)
    gam = 0.99
    alpha = 0.5
    lamb = 0.9
    agent = TDLambdaAgent(env, gamma=gam, alpha=alpha, lamb=lamb)
    method_label = f'TD(Lambda={lamb})'
    method_label = f"{method_label} (gamma={gam}, alpha={alpha})"
    keyboard_play(env, agent, method_label=method_label)
    env.close()
