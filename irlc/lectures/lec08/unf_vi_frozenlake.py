# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.gridworld.gridworld_environments import FrozenLake
from irlc.ex01.agent import train
from irlc.gridworld.demo_agents.hidden_agents import ValueIterationAgent3
from irlc import interactive

def q1_vi(env):
    agent = ValueIterationAgent3(env, epsilon=0, gamma=1, only_update_current=False)
    env, agent = interactive(env, agent)
    env.reset()
    train(env, agent, num_episodes=100)
    env.close()


if __name__ == "__main__":
    env = FrozenLake(render_mode='human', living_reward=-0)
    q1_vi(env)
