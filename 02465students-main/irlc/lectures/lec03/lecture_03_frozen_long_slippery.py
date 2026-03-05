# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.chapter1.dp_planning_agent import dp_visualization
from irlc.gridworld.gridworld_environments import FrozenLake

if __name__ == "__main__":
    env = FrozenLake(is_slippery=True, living_reward=-1e-4, render_mode='human')
    dp_visualization(env, N=40, num_episodes=100)
    env.close()
