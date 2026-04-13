    deltas = []  
    for t, (s, r) in enumerate(zip(states[:-1], rewards)):
        sp = states[t + 1]
        delta = (r + gamma * v[sp]) - v[s]
        deltas.append(delta)  