from irlc.exam.exam2024spring.mdp import MDP
from irlc.exam.exam2024spring.policy_evaluation import policy_evaluation
from irlc.exam.exam2024spring.value_iteration import value_iteration

class BigSpender(MDP): 
    def __init__(self, r_airbnb=0.01):
        self.p_win = 0.45
        self.r_airbnb = r_airbnb
        super().__init__(initial_state=1) # s0 = 1 means we have an appartment.

    def is_terminal(self, state):
        return False

    def A(self, s):
        if s == 0: # if there is no appartment, there is nothing we can do
            return [0]
        if s == 1: # If we have an appartment, we can airbnb, a=0, or gamble, a=1.
            return [0, 1]

    def Psr(self, s, a):
        if s == 0:
            return {(0, 0): 1} # No appartment means p(s=0, r=0 | s,a) = 1.
        if s == 1 and a == 1: # with appartment and gambling
            return {(0, 0): 1-self.p_win,  # p(s=0, r=0 | s,a=1) = 1-p_win
                    (1, 2): self.p_win}  # p(s=1, r=2 | s,a=1) = p_win
        if s == 1 and a == 0: # with appartment and no gambling, p(s=1, r=r_airbnb | s,a) = 1.
            return {(1, self.r_airbnb): 1} 

def a_always_airbnb(r_airbnb : float, gamma : float) -> float:
    mdp = BigSpender(r_airbnb=r_airbnb) 
    pi = {0: {0: 1},
          1: {0: 1, 1:0}}
    J = policy_evaluation(pi=pi, mdp=mdp, gamma=gamma)
    r1 = mdp.r_airbnb * 1/(1-gamma) #n.b. this solution, which simply compute the return explicitly, is also legal.
    r2 = J[1]
    assert abs(r1 - r2) < 1e-3
    v = r1 
    return v

def b_random_decisions(r_airbnb : float, gamma : float) -> float:
    mdp = BigSpender(r_airbnb=r_airbnb) 
    pi = {0: {0: 1}, 1: {0: 0.5, 1: 0.5}}
    J = policy_evaluation(pi=pi, mdp=mdp, gamma=gamma)
    v = J[1] 
    return v

def c_is_it_better_to_gamble(r_airbnb : float, gamma : float) -> bool:
    mdp = BigSpender(r_airbnb=r_airbnb) 
    pi, V = value_iteration(mdp, gamma)
    better_to_gamble = pi[1] == 1 
    return better_to_gamble

if __name__ == "__main__":
    print("a) The expected return is approximately 1, your result:", a_always_airbnb(r_airbnb=0.01, gamma=0.99))
    print("b) The expected return is approximately 1.612, your result:", b_random_decisions(r_airbnb=0.01, gamma=0.99))
    print("c1) In this case, you should return False as it is better to AirBnB, your result:", c_is_it_better_to_gamble(r_airbnb=0.02, gamma=0.99))
    print("c2) In this case, you should return True as it is better to gamble, your result:", c_is_it_better_to_gamble(r_airbnb=0.01, gamma=0.99))