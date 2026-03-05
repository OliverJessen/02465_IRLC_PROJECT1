# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.gridworld.gridworld_environments import BookGridEnvironment
from irlc.lectures.unf.unf_policy_evaluation_gridworld import policy_improvement

if __name__ == "__main__":
    env = BookGridEnvironment(render_mode='human', living_reward=-0.05)
    policy_improvement(env)
