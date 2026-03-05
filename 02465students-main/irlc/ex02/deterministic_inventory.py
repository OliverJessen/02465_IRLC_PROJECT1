# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex02.dp_model import DPModel

class DeterministicInventoryDPModel(DPModel):
    def __init__(self, N=3):
        super().__init__(N=N)

    def A(self, x, k): # Action space A_k(x)
        return {0, 1, 2}

    def S(self, k): # State space S_k
        return {0, 1, 2}

    def g(self, x, u, w, k): # Cost function g_k(x,u,w)
        return u + (x + u - w) ** 2

    def f(self, x, u, w, k): # Dynamics f_k(x,u,w)
        return max(0, min(2, x + u - w ))

    def Pw(self, x, u, k): # Distribution over random disturbances 
        """In this problem we assume that p(w=k+1) = 1.
        Return this as a dictionary of the form: {w : p(w)}."""
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def gN(self, x):
        return 0


def main():
    model = DeterministicInventoryDPModel()
    x = 1
    u = 1
    k = 1
    p_w = model.Pw(x, u, k)
    print("Probability that w=k+1 is p(w=k+1) =", p_w[k+1], "(should be 1)")


if __name__ == "__main__":
    main()
