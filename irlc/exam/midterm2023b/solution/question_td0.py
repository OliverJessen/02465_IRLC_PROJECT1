def a_compute_deltas(v: dict, states: list, rewards: list, gamma: float) -> list:
    deltas = []  # !b;nolines
    for t, (s, r) in enumerate(zip(states[:-1], rewards)):
        sp = states[t + 1]
        delta = (r + gamma * v[sp]) - v[s]
        deltas.append(delta)  # !b
    return deltas


def b_perform_td0(v: dict, states: list, rewards: list, gamma: float, alpha: float) -> dict:
    for t in range(len(rewards)):  # !b;nolines
        s = states[t]
        sp = states[t + 1]
        r = rewards[t]
        delta = r + gamma * v[sp] - v[s]
        v[s] = v[s] + alpha * delta  # !b
    return v


def c_perform_td0_batched(v: dict, states: list, rewards: list, gamma: float, alpha: float) -> dict:
    deltas = a_compute_deltas(v, states, rewards, gamma)  # !b;nolines
    for t in range(len(rewards)):
        s = states[t]
        v[s] = v[s] + alpha * deltas[t]  # !b
    return v


if __name__ == "__main__":
    states = [1, 0, 2, -1, 2, 4, 5, 4, 3, 2, 1, -1]
    rewards = [1, 0.5, -1, 0, 1, 2, 2, 0, 0, -1, 0.5]
    # In the notation of the problem: T = len(rewards).
    v = {s: 0 for s in states}  # Initialize the value function v.
    gamma = 0.9
    alpha = 0.2

    deltas = a_compute_deltas(v, states, rewards, gamma)
    print(f"The first value of delta should be 1, your value is {deltas[0]=}")

    v = b_perform_td0(v, states, rewards, gamma, alpha)
    print(f"The value function v(s=1) should be 0.25352, your value is {v[1]=}")

    v_batched = {s: 0 for s in states}  # Initialize the value function anew
    v_batched = c_perform_td0_batched(v_batched, states, rewards, gamma, alpha)
    print(f"The batched value function in v(s=1) should be 0.3, your value is {v_batched[1]=}")