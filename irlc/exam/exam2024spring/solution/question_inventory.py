import math
from irlc.exam.exam2024spring.inventory import InventoryDPModel
from irlc.exam.exam2024spring.dp import DP_stochastic

class InventoryDPModelGowns(InventoryDPModel): 
    action_sale = "sale"
    def __init__(self, N=3, m=3, allow_sale=False):
        self.m = m
        self.allow_sale = allow_sale
        super().__init__(N=N)

    def A(self, x, k):  # Action space A_k(x)
        space = list(range(self.m))
        if self.allow_sale:
            space = space + [self.action_sale]
            return space
        else:
            return space

    def g(self, x, u, w, k):  # Cost function g_k(x,u,w)
        if u == self.action_sale:
            return 3/4 * (self.m - w)
        else:
            return InventoryDPModel.g(self, x, u, w, k)

    def f(self, x, u, w, k):  # Dynamics f_k(x,u,w)
        if u == self.action_sale:
            return 0
        else:
            return InventoryDPModel.f(self, x, u, w, k)  # max(0, min(self.m, x + u - w))

    def Pw(self, x, u, k):  # Distribution over random disturbances
        pw = {w: 1/self.m for w in range(self.m)}
        assert math.fabs(sum(pw.values())  - 1) < 1e-6
        return pw

def a_get_cost(N: int, m: int, x0 : int) -> float:
    model = InventoryDPModelGowns(N=N, m=m, allow_sale=False) 
    J, pi = DP_stochastic(model)
    expected_cost = J[0][x0] 
    return expected_cost

def b_sale(N : int, m : int, x0 : int) -> float:
    model = InventoryDPModelGowns(N=N, m=m,  allow_sale=True) 
    J, pi = DP_stochastic(model)
    expected_cost = J[0][x0] 
    return expected_cost


if __name__ == "__main__":
    x0 = 0
    N = 6
    m = 4
    print(f"a) The expected cost should be 13.75, and you got {a_get_cost(N, m=m, x0=x0)=}")
    print(f"b) Expected cost when the sales-option is available should be approximately 11.25, and you got {b_sale(N, m=m, x0=x0)=}")