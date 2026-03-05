# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.gridworld.gridworld_environments import BookGridEnvironment
from irlc.lectures.lec11.sarsa_nstep_delay import SarsaDelayNAgent
from irlc import interactive, train

if __name__ == "__main__":
    env = BookGridEnvironment(render_mode='human')
    agent = SarsaDelayNAgent(env, gamma=1, epsilon=0.1, alpha=0.9, n=1) # Exam problem.
    # agent = SarsaDelayNAgent(env, gamma=0.95, epsilon=0.1, alpha=.2, n=1)
    env, agent = interactive(env, agent)
    train(env, agent, num_episodes=10)
