from irlc.exam.midterm2023a.inventory import InventoryDPModel

def a_expected_items_next_day(x : int, u : int) -> float:
    model = InventoryDPModel()
    expected_number_of_items = None
    k = 0 
    expected_number_of_items = sum([p * model.f(x, u, w, k=0) for w, p in model.Pw(x, u, k).items()]) 
    return expected_number_of_items


def b_evaluate_policy(pi : list, x0 : int) -> float:
    model = InventoryDPModel()     
    N = model.N
    J = [{} for _ in range(N + 1)]
    J[N] = {x: model.gN(x) for x in model.S(model.N)}
    for k in range(N - 1, -1, -1):
        for x in model.S(k):
            Qu = {u: sum(pw * (model.g(x, u, w, k) + J[k + 1][model.f(x, u, w, k)]) for w, pw in model.Pw(x, u, k).items()) for u
                  in model.A(x, k)}

            umin = pi[k][x] # min(Qu, key=Qu.get)
            J[k][x] = Qu[umin]  # Compute the expected cost function
    J_pi_x0 = J[0][x0] 
    return J_pi_x0

if __name__ == "__main__":
    model = InventoryDPModel()
    # Create a policy that always buy an item if the inventory is empty.
    pi = [{s: 1 if s == 0 else 0 for s in model.S(k)} for k in range(model.N)]
    x = 0
    u = 1
    x0 = 1
    a_expected_items_next_day(x=0, u=1)
    print(f"Given inventory is {x=} and we buy {u=}, the expected items on day k=1 is {a_expected_items_next_day(x, u)} and should be 0.1")
    print(f"Evaluation of policy is {b_evaluate_policy(pi, x0)} and should be 2.7")