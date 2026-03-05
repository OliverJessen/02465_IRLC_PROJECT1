# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex11.semi_grad_q import LinearSemiGradQAgent
from irlc.pacman.pacman_environment import PacmanEnvironment
from irlc.ex01.agent import train
from irlc import interactive
from irlc.lectures.chapter14lectures.lecture11pacman import layout, rns
# from irlc import VideoMonitor

if __name__ == "__main__":
    env = PacmanEnvironment(animate_movement=False, layout=layout)

    n, agent = rns[-1]
    agent = agent(env)
    # env, agent = interactive(env, agent)

    train(env, agent, num_episodes=100, max_runs=20)
    env2 = PacmanEnvironment(animate_movement=True, layout=layout, render_mode='human')
    # agent.env = env2
    env2, agent = interactive(env2, agent)
    train(env2, agent, num_episodes=100, max_runs=20)
    env2.close()
