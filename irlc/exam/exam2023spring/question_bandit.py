import numpy as np

def a_select_next_action_epsilon0(k : int, actions : list, rewards : list) -> int:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return a

def b_select_next_action(k : int, actions : list, rewards : list, epsilon : float) -> int:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return a

def c_nonstationary_Qs(k : int, actions : list, rewards : list, alpha : float) -> dict:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return Q

if __name__ == "__main__":
    actions =  [1, 0, 2, 1, 2, 4, 5, 4, 3, 2, 1, 1]
    rewards = [1, 1, 1, 0, 1, 3, 2, 0, 4, 1, 1, 2]
    k = 10

    a_t = a_select_next_action_epsilon0(k, actions, rewards)
    print(f"a) The next action is suppoed to be 3, you computed {a_t}")
    print(f"b) The action you computed was",  b_select_next_action(k, actions, rewards, epsilon=0.3))
    Q = c_nonstationary_Qs(k, actions, rewards, alpha=0.1)
    print(f"c) The Q-value associated with arm a=2 is supposed to be Q(2) = 0.271, you got", Q[2])
