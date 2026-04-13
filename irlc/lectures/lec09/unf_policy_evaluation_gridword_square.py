# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment
from irlc.lectures.lec09.unf_policy_evaluation_gridworld import policy_evaluation

if __name__ == "__main__":
    env = SuttonCornerGridEnvironment(render_mode='human', living_reward=1)
    policy_evaluation(env, gamma=0.9)
