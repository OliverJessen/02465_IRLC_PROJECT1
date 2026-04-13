# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import matplotlib.pyplot as plt
import numpy as np


def pi_optimal(s : int) -> int: 
    """ Compute the optimal policy for Jar-Jar binks. Don't overthink this one! """
    # TODO: 1 lines missing.
    raise NotImplementedError("Return the optimal action in state s.")
    return action

def Q0_approximate(gamma : float, N : int) -> float: 
    """ Return the (estimate) of the optimal action-value function Q^*(0,1) based on
    the first N rewards using a discount factor of gamma. Note the similarity to the n-step estimator. """
    # TODO: 1 lines missing.
    raise NotImplementedError("Return N-term approximation of the optimal action-value function Q^*(0,1)")
    return return_estimate

def Q_exact(s : int,a : int, gamma : float) -> float:
    """
    Return the exact optimal action-value function Q^*(s,a) in the Jar-Jar problem.
    I recommend focusing on simple cases first, such as the two cases in the problem.
    Then try to look at larger values of s (for instance, s=2), first using actions that 'point in the right direction' (a = -1)
    and then actions that point in the 'wrong' direction a=1.

    There are several ways to solve the problem, but the simplest is probably to use recursions.

    *Don't* use your solution to Q0_approximate; it is an approximate (finite-horizon) approximation.
    """
    # TODO: 6 lines missing.
    raise NotImplementedError("return optimal action-value function Q^*(s,a) as a float.")


if __name__ == "__main__":
    gamma = 0.8

    ss = np.asarray(range(-10, 10))
    # Make a plot of your (exact) action-value function Q(s,-1) and Q(s,1).
    plt.plot(ss, [Q_exact(s, -1, gamma) for s in ss], 'k-', label='Exact, a=-1')
    plt.plot(ss, [Q_exact(s, 1, gamma) for s in ss], 'r-', label='Exact, a=1')
    plt.legend()
    plt.grid()
    plt.show()
    print("All done")
