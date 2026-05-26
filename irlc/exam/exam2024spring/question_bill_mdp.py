from irlc.exam.exam2024spring.mdp import MDP
from irlc.exam.exam2024spring.policy_evaluation import policy_evaluation
from irlc.exam.exam2024spring.value_iteration import value_iteration

# TODO: Code has been removed from here.
raise NotImplementedError("Insert your solution and remove this error.")

def a_always_airbnb(r_airbnb : float, gamma : float) -> float:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return v

def b_random_decisions(r_airbnb : float, gamma : float) -> float:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return v

def c_is_it_better_to_gamble(r_airbnb : float, gamma : float) -> bool:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return better_to_gamble

if __name__ == "__main__":
    print("a) The expected return is approximately 1, your result:", a_always_airbnb(r_airbnb=0.01, gamma=0.99))
    print("b) The expected return is approximately 1.612, your result:", b_random_decisions(r_airbnb=0.01, gamma=0.99))
    print("c1) In this case, you should return False as it is better to AirBnB, your result:", c_is_it_better_to_gamble(r_airbnb=0.02, gamma=0.99))
    print("c2) In this case, you should return True as it is better to gamble, your result:", c_is_it_better_to_gamble(r_airbnb=0.01, gamma=0.99))
