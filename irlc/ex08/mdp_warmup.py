# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (Freely available online).
"""
from irlc.ex08.mdp import MDP


def value_function2q_function(mdp : MDP, s, gamma, v : dict) -> dict: 
    r"""This helper function converts a value function to an action-value function.

    Given a value-function ``v`` and a state ``s``, this function implements the update:

    .. math::

        Q(s,a) = \mathbb{E}[r + \gamma * v(s') | s, a] = \sum_{r, s'} (r + \gamma v(s') ) p(s', r| s,a)

    as described in (SB18, ). It should return a dictionary of the form::

        {a1: Q(s,a1), a2: Q(s,a2), ..., an: Q(s,an)}

    where the actions are keys. You can compute these using ``mdp.A(s)``. When done the following should work::

        Qs = value_function2q_function(mdp, s, gamma, v)
        Qs[a] # This is the Q-value Q(s,a)

    Hints:

        * Remember that ``v[s'] = 0`` if ``s'`` is a terminal state (this is explained in (SB18)).

    :param mdp: An MDP instance. Use this to compute :math:`p(s', r| s,a)`
    :param s: A state
    :param gamma: The discount factor :math:`\gamma`
    :param v: The value function represented as a dictionary.
    :return: A dictionary representing :math:`Q` of the form ``{a1: Q(s,a1), a2: Q(s,a2), ..., an: Q(s,an)}``
    """
    # TODO: 1 lines missing.
    # TODO: 1 lines missing.
    raise NotImplementedError("Implement function body")
    return q_dict

def expected_reward(mdp : MDP, s, a) -> float:
    # TODO: 1 lines missing.
    raise NotImplementedError("Insert your solution and remove this error.")
    return expected_reward

def q_function2value_function(policy : dict, Q : dict, s) -> float:
    # TODO: 1 lines missing.
    raise NotImplementedError("Insert your solution and remove this error.")
    return V_s

if __name__ == "__main__":
    from irlc.gridworld.gridworld_environments import FrozenLake
    mdp = FrozenLake(living_reward=0.2).mdp # Get the MDP of this environment.

    ## Part 1: Expected reward
    s0 = mdp.initial_state 
    s0 = (0, 3) #  initial state
    a = 3 # Go east.
    print("Expected reward E[r | s0, a] =", expected_reward(mdp, s=s0, a=0), "should be 0.2")
    print("Expected reward E[r | s0, a] =", expected_reward(mdp, s=(1, 2), a=0), "should be 0") 


    ## Part 2
    # First let's create a non-trivial value function
    V = {} 
    for s in mdp.nonterminal_states:
        V[s] = s[0] + 2*s[1]
    print("Value function is", V)
    # Compute the corresponding Q(s,a)-values in state s0:
    q_ = value_function2q_function(mdp, s=s0, gamma=0.9, v=V)
    print(f"Q-values in {s0=} is", q_) 

    ## Part 3
    # Create a non-trivial Q-function for this problem.
    Q = {} 
    for s in mdp.nonterminal_states:
        for a in mdp.A(s):
            Q[s,a] = s[0] + 2*s[1] - 10*a # The particular values are not important in this example
    # Create a policy. In this case pi(a=3) = 0.4.
    pi = {0: 0.2,
          1: 0.2,
          2: 0.2,
          3: 0.4}
    print(f"Value-function in {s0=} is", q_function2value_function(pi, Q, s=s0)) 
