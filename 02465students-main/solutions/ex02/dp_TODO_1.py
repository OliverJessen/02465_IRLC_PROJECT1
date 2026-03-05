            Qu = {u: sum(pw * (model.g(x, u, w, k) + J[k + 1][model.f(x, u, w, k)]) for w, pw in model.Pw(x, u, k).items()) for u in model.A(x, k)} 
            umin = min(Qu, key=Qu.get)
            J[k][x] = Qu[umin] # Compute the expected cost function
            pi[k][x] = umin # Compute the optimal policy 