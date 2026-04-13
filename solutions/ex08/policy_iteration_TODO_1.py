        for s in [mdp.nonterminal_states[i] for i in np.random.permutation(len(mdp.nonterminal_states))]:  
            old_a = pi[s] # The best action we would take under the current policy
            Qs = value_function2q_function(mdp, s, gamma, V)
            pi[s] = max(Qs, key=Qs.get)
            if old_a != pi[s]:
                policy_stable = False 