# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex01.agent import Agent
from irlc.gridworld.gridworld_environments import FrozenLake
from irlc import interactive, train

if __name__ == "__main__":
    env = FrozenLake(render_mode='human', print_states=True)
    env, agent = interactive(env, Agent(env))
    agent.label = "Random agent"
    train(env, agent, num_episodes=100, verbose=False)
    env.close()
