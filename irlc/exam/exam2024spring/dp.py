# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.
"""
from irlc.ex02.dp_model import DPModel

def DP_stochastic(model: DPModel): 
    """
    Implement the stochastic DP algorithm. The implementation follows (Her21, Algorithm 1).
    Once you are done, you should be able to call the function as:

    .. runblock:: pycon

        >>> from irlc.ex02.graph_traversal import SmallGraphDP
        >>> model = SmallGraphDP(t=5)  # Instantiate the small graph with target node 5
        >>> J, pi = DP_stochastic(model)
        >>> print(pi[0][2]) # Action taken in state ``x=2`` at time step ``k=0``.

    :param model: An instance of :class:`irlc.ex01.dp_model.DPModel` class. This represents the problem we wish to solve.
    :return:
        - ``J`` - A list of of cost function so that ``J[k][x]`` represents :math:`J_k(x)`
        - ``pi`` - A list of dictionaries so that ``pi[k][x]`` represents :math:`\mu_k(x)`
    """

    """ 
    In case you run into problems, I recommend following the hints in (Her21, Subsection 6.2.1) and focus on the
    case without a noise term; once it works, you can add the w-terms. When you don't loop over noise terms, just specify
    them as w = None in env.f and env.g.
    """
    N = model.N
    J = [{} for _ in range(N + 1)]
    pi = [{} for _ in range(N)]
    J[N] = {x: model.gN(x) for x in model.S(model.N)}
    for k in range(N-1, -1, -1):
        for x in model.S(k):
            """
            Update pi[k][x] and Jstar[k][x] using the general DP algorithm given in (Her21, Algorithm 1).
            If you implement it using the pseudo-code, I recommend you define Q as a dictionary like the J-function such that
                        
            > Q[u] = Q_u (for all u in model.A(x,k))
            Then you find the u where Q_u is lowest, i.e. 
            > umin = arg_min_u Q[u]
            Then you can use this to update J[k][x] = Q_umin and pi[k][x] = umin.
            """
            Qu = {u: sum(pw * (model.g(x, u, w, k) + J[k + 1][model.f(x, u, w, k)]) for w, pw in model.Pw(x, u, k).items()) for u in model.A(x, k)}
            umin = min(Qu, key=Qu.get)
            J[k][x] = Qu[umin] # Compute the expected cost function
            pi[k][x] = umin # Compute the optimal policy
            """
            After the above update it should be the case that:

            J[k][x] = J_k(x)
            pi[k][x] = pi_k(x)
            """
    return J, pi 
