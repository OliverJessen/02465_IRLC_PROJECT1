# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.gridworld.gridworld_environments import BookGridEnvironment
from irlc import interactive, train
from irlc.gridworld.demo_agents.hidden_agents import PolicyEvaluationAgent2

def policy_evaluation_stepwise(env=None):
    agent = PolicyEvaluationAgent2(env, gamma=1., steps_between_policy_improvement=None, only_update_current=True)
    env, agent = interactive(env, agent)
    train(env, agent, num_episodes=100)
    env.close()

def policy_improvement(env=None, q_mode=True):
    agent = PolicyEvaluationAgent2(env, gamma=1.,steps_between_policy_improvement=20)
    env, agent = interactive(env, agent)
    train(env, agent, num_episodes=1000)
    env.close()

if __name__ == "__main__":
    env = BookGridEnvironment(render_mode='human', living_reward=-0.05)
    policy_evaluation_stepwise(env)
