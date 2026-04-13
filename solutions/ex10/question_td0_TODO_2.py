    for t in range(len(rewards)):  
        s = states[t]
        sp = states[t + 1]
        r = rewards[t]
        delta = r + gamma * v[sp] - v[s]
        v[s] = v[s] + alpha * delta  