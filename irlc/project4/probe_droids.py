# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from irlc import train
from irlc.gridworld.gridworld_environments import GridworldEnvironment
from irlc.ex11.feature_encoder import FeatureEncoder
from irlc.ex11.semi_grad_sarsa import LinearSemiGradSarsa
from irlc.ex10.sarsa_agent import SarsaAgent
from irlc import interactive, savepdf

np.seterr(all='raise')

small_circle_grid = [[' ', +1,'#'],
                     [' ','#', -1],
                     ['S',' ',' ']]

class TinyGridworld(GridworldEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(small_circle_grid, *args, **kwargs)

# TODO: Code has been removed from here. 

def linear_experiment(episodes=10000, alpha=0.02, states_and_actions=( ((0,0),0) )):
    env = TinyGridworld()
    # TODO: Code has been removed from here.
    raise NotImplementedError("Your implementation here.")
    q_values = {(s, a): agent.Q(s, a) for s, a in states_and_actions} # Example: If you trained an agent, this will extract the q-values.
    return q_values

def quadratic_experiment(episodes=10000, alpha=0.02, states_and_actions=( ((0,0),0) )):
    env = TinyGridworld()
    # TODO: Code has been removed from here.
    raise NotImplementedError("Your code here")
    q_values = {(s, a): agent.Q(s, a) for s, a in states_and_actions}  # Example: If you trained an agent, this will extract the q-values.
    return q_values


if __name__ == "__main__":
    env = TinyGridworld()
    agent = SarsaAgent(env, alpha=0.02, gamma=1., epsilon=1) 
    train(env, agent, num_episodes=10000, verbose=False) 
    # Example: This will extract the states/actions.
    agent.label = "Sarsa after 10000 episodes, alpha=0.02"
    env, agent = interactive(env, agent)
    savepdf('sarsa_eval10k', env=env)
    env.close()
    # Evaluate the Sarsa agent:
    states_and_actions = [((0, 0), 0),
                          ((0, 0), 1),
                          ((2, 0), 3),
                          ((0, 2), 1),
                          ]
    q_values = {(s, a): agent.Q[s, a] for s, a in states_and_actions}
    for (s, a), q in q_values.items(): 
        print(f"In state {s=} and action {a=} we have Q(s,a) = {q}.")
    # Question 1: Estimate using the linear function approximator
    qs = linear_experiment(episodes=10000, alpha=0.01, states_and_actions=states_and_actions)
    for (s,a), q in qs.items(): 
        print(f"In state {s=} and action {a=} we have Q(s,a)={q} (Linear function approximator)")
    # Question 2: Estimate using the quadratic function approximator
    qs = quadratic_experiment(episodes=10000, alpha=0.01, states_and_actions=states_and_actions)
    for (s, a), q in qs.items(): 
        print(f"In state {s=} and action {a=} we have Q(s,a)={q} (Quadratic function approximator)")
