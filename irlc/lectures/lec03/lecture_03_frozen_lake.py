# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.gridworld.gridworld_environments import FrozenLake
from gymnasium.wrappers import TimeLimit
from irlc import Agent, interactive, train

if __name__ == "__main__":
    env = FrozenLake(is_slippery=True, living_reward=-1e-4, render_mode="human")
    N = 40
    env, agent = interactive(env, Agent(env))
    env = TimeLimit(env, max_episode_steps=N)
    num_episodes = 100
    train(env, agent, num_episodes=num_episodes)
    env.close()
