            q = value_function2q_function(mdp, s, gamma, v) 
            v_, v[s] = v[s], sum( [q[a] * pi_a for a,pi_a in pi[s].items()] ) 