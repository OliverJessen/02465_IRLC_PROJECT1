from irlc.exam.exam2025spring.inventory import InventoryDPModel
from irlc.exam.exam2025spring.dp import DP_stochastic


# TODO: Code has been removed from here. 

def a_expected_cost(x0 : int, u0 : int) -> float:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return expected_cost

def b_best_action(N : int, cost_per_cake : float, k : int, x : int) -> int:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return best_action

def c_lazy_baker(N : int, cost_per_cake : float, x0 : int) -> float:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return cost

if __name__ == "__main__":
    print(f"a) The expected cost should be 1.3 and you got {a_expected_cost(x0=0, u0=1)=}")
    print(f"b) Using the modified cost the best action is 1 and you got: {b_best_action(N=3, cost_per_cake=0.8, k=0, x=1)=}")
    print(f"c) The expected cost for the lazy baker is approximately 1.311 and you got: {c_lazy_baker(N=3, cost_per_cake=0.7, x0=0)=}")
