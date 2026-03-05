# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex04.pid_lunar import get_lunar_lander
import gymnasium
from irlc import train

if __name__ == "__main__":
    env = gymnasium.make('LunarLanderContinuous-v3', render_mode='human')
    env._max_episode_steps = 1000  # We don't want it to time out.

    agent = get_lunar_lander(env)
    stats, traj = train(env, agent, return_trajectory=True, num_episodes=10)
    env.close()
