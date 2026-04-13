# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (Freely available online).
"""
import numpy as np
from irlc.ex08.small_gridworld import SmallGridworldMDP
import matplotlib.pyplot as plt
from irlc.ex08.policy_evaluation import policy_evaluation
from irlc.ex08.mdp_warmup import value_function2q_function

def policy_iteration(mdp, gamma=1.0):
    r"""
    Implement policy iteration (see (SB18, Section 4.3)).

    Note that policy iteration only considers deterministic policies. we will therefore use the shortcut by representing the policy pi
    as a dictionary (similar to the DP-problem in week 2!) so that
    > a = pi[s]
    is the action in state s.

    """
    pi = {s: np.random.choice(mdp.A(s)) for s in mdp.nonterminal_states}
    policy_stable = False
    V = None # Sutton has an initialization-step, but it can actually be skipped if we intialize the policy randomly.
    while not policy_stable:
        # Evaluate the current policy using your code from the previous exercise.
        # The main complication is that we need to transform our deterministic policy, pi[s], into a stochastic one pi[s][a].
        # It will be defined as:
        # >>>  pi_prob[s][a] = 1 if a = pi[s] and otherwise 0.
        pi_prob = {s: {a: 1 if pi[s] == a else 0 for a in mdp.A(s)} for s in mdp.nonterminal_states}
        V = policy_evaluation(pi_prob, mdp, gamma)
        V = policy_evaluation( {s: {pi[s]: 1} for s in mdp.nonterminal_states}, mdp, gamma)
        r""" Implement the method. This is step (3) in (SB18). """
        policy_stable = True   # Will be set to False if the policy pi changes
        r""" Implement the steps for policy improvement here. Start by writing a for-loop over all non-terminal states
        you can see the policy_evaluation function for how to do this, but 
        I recommend looking at the property mdp.nonterminal_states (see MDP class for more information). 
        Hints:
            * In the algorithm in (SB18), you need to perform an argmax_a over what is actually Q-values. The function
            value_function2q_function(mdp, s, gamma, V) can compute these. 
            * The argmax itself, assuming you follow the above procedure, involves a dictionary. It can be computed 
            using methods similar to those we saw in week2 of the DP problem.
            It is not a coincidence these algorithms are very similar -- if you think about it, the maximization step closely resembles the DP algorithm!
        """
        # TODO: 6 lines missing.
        raise NotImplementedError("Insert your solution and remove this error.")
    return pi, V

if __name__ == "__main__":
    mdp = SmallGridworldMDP()
    pi, v = policy_iteration(mdp, gamma=0.99)
    expected_v = np.array([ 0, -1, -2, -3,
                           -1, -2, -3, -2,
                           -2, -3, -2, -1,
                           -3, -2, -1,  0])

    from irlc.ex08.small_gridworld import plot_value_function
    plot_value_function(mdp, v)
    plt.title("Value function using policy iteration to find optimal policy")
    from irlc import savepdf
    savepdf("policy_iteration")
    plt.show()
