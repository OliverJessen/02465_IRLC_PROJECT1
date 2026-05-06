def a_greedy_policy(q_values : dict, state : int) -> int:
    qs = {a : q_values[state, a] for a in [0, 1]} 
    astar = max(qs, key=qs.get)
    # alternative solution:
    q0 = q_values[state, 0] if (state, 0) in q_values else 0
    q1 = q_values[state, 1] if (state, 1) in q_values else 0
    if q0 > q1:
        astar_alt = 0
    else:
        astar_alt = 1
    assert astar_alt == astar  
    return astar

def b_update_single_q(alpha, gamma, q_values: dict, state : int, action : int, reward : float, next_state : int) -> float:
    q_before = q_values[state, action] 
    updated_q = q_before + alpha * (reward + gamma * max( [ q_values[next_state, a] for a in [0, 1] ]) - q_before ) 
    return updated_q

def c_update_all_q(alpha, gamma, states_actions_rewards: list[tuple]) -> dict:
    q2 = dict() 
    for t, (state, action, reward) in enumerate(states_actions_rewards[:-1]):
        next_state, _, _ = states_actions_rewards[t + 1]
        if (state, action) not in q2:
            q2[state, action] = 0
        for a in [0, 1]:
            if (next_state, a) not in q2:
                q2[next_state, a] = 0

        q2[state, action] = q2[state, action] + alpha * (reward + gamma * max( [ q2[next_state, a] for a in [0, 1] ]) - q2[state, action] )
    q_values = q2 
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