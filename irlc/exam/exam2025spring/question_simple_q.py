def a_greedy_policy(q_values : dict, state : int) -> int:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return astar

def b_update_single_q(alpha, gamma, q_values: dict, state : int, action : int, reward : float, next_state : int) -> float:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return updated_q

def c_update_all_q(alpha, gamma, states_actions_rewards: list[tuple]) -> dict:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return q_values


if __name__ == "__main__":
    # Example of Q-values:
    states = [0, 1, 2]
    actions = [0, 1]
    q_example = {} # Initialize a small example of Q-values.
    for s in states:
        for a in actions:
            q_example[s,a] = s/2 + 2 ** a # Initialize so that Q(s, a) = s / 2 + 2**a

    print(f"a) The greedy action in state s=0. Should be a* = 1, you got {a_greedy_policy(q_example, state=0)=}")

    alpha = 0.8
    gamma = 0.9
    state = 0
    action = 1
    reward = 0.8
    next_state = 2

    print(f"b) Q(0, 1) was {q_example[state, action]=} and should be updated to 3.2. You got {b_update_single_q(alpha, gamma, q_example, state, action, reward, next_state)=}")

    # The trajectory is of the form [..., (S_t, A_t, R_{t+1}), ... ]
    example_trajectory = [(0, 1, 0.5),   # s_0 = 0, a_0 = 1, r_1 = 0.5  
                          (2, 0, -0.75), # s_1 = 2, a_1 = 0, r_2 = -0.75
                          (0, 1, 0.5),   # s_2 = 0, a_2 = 1, r_3 = 0.4
                          (1, 0, 0.5)]   # s_3 = 1, a_3 = 0, r_4 = -0.75   

    updated_q_values = c_update_all_q(alpha, gamma, example_trajectory) # This should be a dictionary.
    print(f"c) Q({state}, {action}) should be updated to 0.48. You got {updated_q_values[state, action]}")
