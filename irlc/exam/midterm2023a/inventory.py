# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
Implements the inventory-control problem from (Her21, Subsection 5.1.2). See todays slides if you are stuck!
References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.
"""
from irlc.exam.midterm2023a.dp_model import DPModel
from irlc.exam.midterm2023a.dp import DP_stochastic

class InventoryDPModel(DPModel): 
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
        return {0:.1, 1:.7, 2:0.2}

    def gN(self, x):
        return 0 

def main():
    inv = InventoryDPModel() 
    J,pi = DP_stochastic(inv)
    print(f"Inventory control optimal policy/value functions")
    for k in range(inv.N):
        print(", ".join([f" J_{k}(x_{k}={i}) = {J[k][i]:.2f}" for i in inv.S(k)] ) )
    for k in range(inv.N):
        print(", ".join([f"pi_{k}(x_{k}={i}) = {pi[k][i]}" for i in inv.S(k)] ) ) 

if __name__ == "__main__":
    main()
