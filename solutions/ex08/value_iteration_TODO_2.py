        Q = {a: v-(1e-8*a if isinstance(a, int) else 0) for a,v in value_function2q_function(mdp, s, gamma, V).items()} 
        pi[s] = max(Q, key=Q.get) 