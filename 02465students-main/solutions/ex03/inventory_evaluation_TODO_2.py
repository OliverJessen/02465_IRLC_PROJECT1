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