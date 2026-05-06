# from irlc.exam.midterm2023b.inventory import InventoryDPModel
# from irlc.exam.midterm2023b.dp import DP_stochastic
# from irlc.exam
# import irlc
import numpy as np
from irlc.exam.midterm2023b.mdp import MDP

class SmallGambler(MDP):
    """
    Implements a variant of the gambler problem. Please refer to the problem text for a description. You can consider this
    implementation of the environment to be authoritative, and I do not recommend changing it.
    """
    def __init__(self):
        goal = 40
        super().__init__(initial_state=goal // 2)
        self.goal = 40
        self.p_heads = .4 # Chance of winning.

    def is_terminal(self, state):
        """ Environment has been modified to never terminate. """
        return False

    def A(self, s):
        """ Action is the amount you choose to gamble.
        You can gamble from 0 and up to the amount of money you have (state),

        If you are either in s = 0 or s = self.goal, you cannot gamble anything (A(s) = {0}). """
        return range(0, min(s, self.goal - s) + 1)

    def Psr(self, s, a):
        """ Implement transition probabilities here.
        the reward is 1 if s < self.goal and s + a == self.goal and otherwise 0. Remember the format should
         return a dictionary with entries:
        > { (sp, r) : probability }
        """
        r = 1 if s + a == self.goal and s < self.goal else -a/100
        if a == 0:
            d = {(s + a, r): 1}
        else:
            d = {(s + a, r): self.p_heads, (s - a, 0): 1 - self.p_heads}
        assert sum(d.values()) == 1  # Sanity check: the probabilities must sum to 1.
        return d


def a_get_reward(s : int, a : int) -> float:
    mdp = SmallGambler()
    avg_reward = 0 
    for (sp, r), p in mdp.Psr(s, a).items():
        avg_reward += r * p 
    return avg_reward

def b_get_best_immediate_action(s : int) -> int:
    mdp = SmallGambler()
    if s not in mdp.nonterminal_states: 
        return 0
    d = {a: a_get_reward(s, a) for a in mdp.A(s)}
    astar = max(d, key=d.get)
    vs = [v for v in d.values() if np.abs(v - d[astar]) < 1e-6]
    if len( vs )>1:
        print(vs)
        assert False   
    return astar

def c_get_best_action_twosteps(s : int) -> int:
    mdp = SmallGambler()
    d = {} 
    for a in mdp.A(s):
        d[a] = 0
        for (sp, r), p in mdp.Psr(s,a).items():
            d[a] += p * (r + a_get_reward(sp, b_get_best_immediate_action(sp)))

    astar = max(d, key=d.get)
    vs = [v for v in d.values() if np.abs(v-d[astar]) < 1e-6]
    if len( vs )>1:
        print(vs)
        assert False          
    return astar

if __name__ == "__main__":
    mdp = SmallGambler()
    s = 16
    a = 26

    print(f"When {s=} and {a=} the average reward is -0.104; your value is {a_get_reward(s,a)=}")
    print(f"When {s=} the best immediate action is 0, your value is {b_get_best_immediate_action(s)=}")
    print(f"When {s=} the best action over two steps is 4, your value is {c_get_best_action_twosteps(s)=}")