# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
def a_compute_deltas(v: dict, states: list, rewards: list, gamma: float) -> list:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return deltas


def b_perform_td0(v: dict, states: list, rewards: list, gamma: float, alpha: float) -> dict:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return v


def c_perform_td0_batched(v: dict, states: list, rewards: list, gamma: float, alpha: float) -> dict:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
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
