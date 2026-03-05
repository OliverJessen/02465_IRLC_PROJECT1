# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex02.inventory import InventoryDPModel
from irlc.ex02.dp import DP_stochastic
import numpy as np

# TODO: Code has been removed from here.
# raise NotImplementedError("Insert your solution and remove this error.")
# We create our own model for the flower store problem, which is a special case of the inventory problem. 
# We will use this model to solve the problem using DP and to compute the probability of ending up with 
# a single element in the inventory at the end of the process.
class FlowerStoreModel(InventoryDPModel): 
    def __init__(self, N=3, c=0., prob_empty=False):
        self.c = c
        self.prob_empty = prob_empty
        super().__init__(N=N)

    def g(self, x, u, w, k):  # New cost function g_k(x,u,w)
        if self.prob_empty:
            return 0
        return u * self.c + np.abs(x + u - w)

    def f(self, x, u, w, k):  # New transition function f_k(x,u,w)
        return max(0, min(max(self.S(k)), x + u - w))

    def Pw(self, x, u, k):  # New distribution over random noise
        pw = {0:.1,1:.3,2:.6}
        return pw

    def gN(self, x): # Terminal cost g_N(x) which we can use for the last part of the problem.
        if self.prob_empty:
            return -1 if x == 1 else 0
        else:
            return 0 


def a_get_policy(N: int, c: float, x0 : int) -> int:
    # TODO: Code has been removed from here.
    model = FlowerStoreModel(N=N, c=c, prob_empty=False) 
    J, pi = DP_stochastic(model)
    u = pi[0][x0]  
    # raise NotImplementedError("Insert your solution and remove this error.")
    return u

def b_prob_one(N : int, x0 : int) -> float:
    # TODO: Code has been removed from here.
    # return the probability that we end up with a single element in the inventory at the end of the process, i.e. P(x_N = 1 | x_0 = x0, pi).
    model = FlowerStoreModel(N=N, prob_empty=True) 
    J, pi = DP_stochastic(model)
    pr_empty = -J[0][x0] 
    # raise NotImplementedError("Insert your solution and remove this error.")
    return pr_empty


if __name__ == "__main__":
    model = InventoryDPModel()
    pi = [{s: 0 for s in model.S(k)} for k in range(model.N)]
    x0 = 0
    c = 0.5
    N = 3
    print(f"a) The policy choice for {c=} is {a_get_policy(N, c,x0)} (should be 1)")
    print(f"b) The probability of ending up with a single element in the inventory is {b_prob_one(N, x0)} and should be 0.492")
