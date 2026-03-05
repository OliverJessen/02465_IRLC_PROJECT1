# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.chapter1.dp_planning_agent import dp_visualization
from irlc.gridworld.gridworld_environments import FrozenLake

if __name__ == "__main__":
    env = FrozenLake(render_mode='human')
    dp_visualization(env, N=4, num_episodes=10)
    env.close()
