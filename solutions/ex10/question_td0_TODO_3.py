    deltas = a_compute_deltas(v, states, rewards, gamma)  
    for t in range(len(rewards)):
        s = states[t]
        v[s] = v[s] + alpha * deltas[t]  