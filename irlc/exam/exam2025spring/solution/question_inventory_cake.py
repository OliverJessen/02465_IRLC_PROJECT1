from irlc.exam.exam2025spring.inventory import InventoryDPModel
from irlc.exam.exam2025spring.dp import DP_stochastic


class CakeInventoryModel(InventoryDPModel): 
    def __init__(self, N=3, cost_per_cake=1., lazybaker=False):
        self.lazybaker = lazybaker
        self.cost_per_cake = cost_per_cake
        super().__init__(N=N)

    def A(self, x, k):  # Action space A_k(x)
        if self.lazybaker:
            return {2 if k % 2 == 0 else 0}
        else:
            return {0, 1, 2}

    def g(self, x, u, w, k): # Cost function g_k(x,u,w)
        cakes_sold = min(w, x + u)
        return u * self.cost_per_cake - cakes_sold

    def gN(self, x):
        return x**2 

def a_expected_cost(x0 : int, u0 : int) -> float:
    model = InventoryDPModel() 
    expected_cost = sum( [model.g(x0, u0, w, k=0) * pw for w, pw in model.Pw(x0, u0, k=0).items()] )
    # The above solution is perhaps more general since it can be adapted to any cost very easily. However,
    # the following explicit solution is more direct:
    expected_cost_2 = u0 + 1/10 * (x0 - 0 + u0)**2  + 7/10 * (x0 - 1 + u0)**2 + 1/5 * (x0 - 2 + u0)**2
    # Let's check that the two solutions agre.
    assert abs( expected_cost_2 - expected_cost) < 1e-8 
    return expected_cost

def b_best_action(N : int, cost_per_cake : float, k : int, x : int) -> int:
    model = CakeInventoryModel(N, cost_per_cake=cost_per_cake) 
    J, pi = DP_stochastic(model)
    best_action = pi[k][x] 
    return best_action

def c_lazy_baker(N : int, cost_per_cake : float, x0 : int) -> float:
    model = CakeInventoryModel(N=N, cost_per_cake=cost_per_cake, lazybaker=True) 
    J, pi = DP_stochastic(model)
    cost = J[0][x0] 
    return cost

if __name__ == "__main__":
    print(f"a) The expected cost should be 1.3 and you got {a_expected_cost(x0=0, u0=1)=}")
    print(f"b) Using the modified cost the best action is 1 and you got: {b_best_action(N=3, cost_per_cake=0.8, k=0, x=1)=}")
    print(f"c) The expected cost for the lazy baker is approximately 1.311 and you got: {c_lazy_baker(N=3, cost_per_cake=0.7, x0=0)=}")