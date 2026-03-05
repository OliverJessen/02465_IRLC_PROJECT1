# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from collections import defaultdict
from irlc import train
from irlc.ex02.dp_model import DPModel
from irlc.ex02.dp import DP_stochastic
from irlc.ex02.dp_agent import DynamicalProgrammingAgent
from irlc.pacman.pacman_environment import PacmanEnvironment
from irlc.pacman.gamestate import GameState

east = """ 
%%%%%%%%
% P   .%
%%%%%%%% """ 

east2 = """
%%%%%%%%
%    P.%
%%%%%%%% """

SS2tiny = """
%%%%%%
%.P  %
% GG.%
%%%%%%
"""

SS0tiny = """
%%%%%%
%.P  %
%   .%
%%%%%%
"""

SS1tiny = """
%%%%%%
%.P  %
%  G.%
%%%%%%
"""

datadiscs = """
%%%%%%%
%    .%
%.P%% %
%.   .%
%%%%%%%
"""

# TODO: 30 lines missing.
# raise NotImplementedError("Put your own code here")

def p_next(x : GameState, u: str): # DONE
    """ Given the agent is in GameState x and takes action u, the game will transition to a new state xp.
    The state xp will be random when there are ghosts. This function should return a dictionary of the form

    {..., xp: p, ...}

    of all possible next states xp and their probability -- you need to compute this probability.

    Hints:
        * In the above, xp should be a GameState, and p will be a float. These are generated using the functions in the GameState x.
        * Start simple (zero ghosts). Then make it work with one ghosts, and then finally with any number of ghosts.
        * Remember the ghosts move at random. I.e. if a ghost has 3 available actions, it will choose one with probability 1/3
        * The slightly tricky part is that when there are multiple ghosts, different actions by the individual ghosts may lead to the same final state
        * Check the probabilities sum to 1. This will be your main way of debugging your code and catching issues relating to the previous point.
    """
    # TODO: 8 lines missing.

    xp = x.f(u)
    if xp.is_won() or xp.is_lost() or xp.player() == 0:
        return {xp: 1.0}
    states = {}
    actions = xp.A()                                        # available actions for the ghosts
    for a in actions:
        for s, p in p_next(xp, a).items():                  # get the next states and their probabilities for each ghost action
            states[s] = states.get(s, 0) + p / len(actions) # uniform probability so divide by len(actions)
    
    # raise NotImplementedError("Return a dictionary {.., xp: p, ..} where xp is a possible next state and p the probability")
    return states

def go_east(map): # DONE
    """ Given a map-string map (see examples in the top of this file) that can be solved by only going east, this will return
    a list of states Pacman will traverse. The list it returns should therefore be of the form:

    [s0, s1, s2, ..., sn]

    where each sk is a GameState object, the first element s0 is the start-configuration (corresponding to that in the Map),
    and the last configuration sn is a won GameState obtained by going east.

    Note this function should work independently of the number of required east-actions.

    Hints:
        * Use the GymPacmanEnvironment class. The report description will contain information about how to set it up, as will pacman_demo.py
        * Use this environment to get the first GameState, then use the recommended functions to go east
    """
    # TODO: 5 lines missing.
    env = PacmanEnvironment(layout_str=map)
    x, _ = env.reset()
    states = [x]                        # Include the initial state
    while not x.is_won(): 
        states.append(x := x.f("East")) # Used the walrus operator (:=) to update x inside append()

    # raise NotImplementedError("Return the list of states pacman will traverse if he goes east until he wins the map")
    return states

def get_future_states(x, N): # DONE
    # TODO: 4 lines missing.

    state_spaces = [[] for _ in range(N+1)]
    state_spaces[0] = [x]
    for k in range(1, N+1):
        state_spaces[k] = list({xp for s in state_spaces[k-1] for a in s.A() for xp in p_next(s, a)}) # remove duplicates by using a set, then convert back to a list

    # raise NotImplementedError("return a list-of-list of future states [S_0, ... ,S_N]. Each S_k is a state space, i.e. a list of GameState objects.")
    return state_spaces

class PacManDPModel(DPModel): # belongs to shortest_path

    def __init__(self, map, N=10):
        env = PacmanEnvironment(layout_str=map)
        x0, _ = env.reset()

        self.x0 = x0
        self.N = N
        self.state_spaces = get_future_states(x0, N)

    def S(self, k):                     # state space at time k
        return self.state_spaces[k]

    def A(self, x, k):                  # actions available in state x at time k
        return x.A()

    def Pw(self, x, u, k):              # transition probabilities based on p_next function
        return p_next(x, u)

    def f(self, x, u, w, k):            # transition function based on DP_Stochastic function
        return w

    def g(self, x, u, w, k):            # cost function
        if w.is_won():
            return 0
        
        if w.is_lost():                
            return 1000

        return 1

    def gN(self, x): # terminal cost function
        return 0

class WinProbabilityModel(PacManDPModel): # inherit from PacManDPModel

    def g(self, x, u, w, k):
        return 0   # no step cost

    def gN(self, x):
        return -1 if x.is_won() else 0

def win_probability(map, N=10): # DONE
    """ Assuming you get a reward of -1 on wining (and otherwise zero), the win probability is -J_pi(x_0). """
    # TODO: 5 lines missing.

    model = WinProbabilityModel(map, N) # reuse same model with different cost function
    J, _ = DP_stochastic(model)
    x0 = model.x0
    win_probability = -J[0][x0]  # win probability is -J_pi(x_0)

    # raise NotImplementedError("Return the chance of winning the given map within N steps or less.")
    return win_probability

def shortest_path(map, N=10): # DONE
    """ If each move has a cost of 1, the shortest path is the path with the lowest cost.
    The actions should be the list of actions taken.
    The states should be a list of states the agent visit. The first should be the initial state and the last
    should be the won state. """
    # TODO: 4 lines missing.

    model = PacManDPModel(map, N)
    J, pi = DP_stochastic(model)
    x, states, actions = model.x0, [model.x0], []
    for k in range(N):
        if x.is_won(): break
        actions.append(u := pi[k][x])
        states.append(x := max(p_next(x,u), key=p_next(x,u).get))

    # raise NotImplementedError("Return the cost of the shortest path, the list of actions taken, and the list of states.")
    return actions, states

def no_ghosts():
    # Check the pacman_demo.py file for help on the GameState class and how to get started.
    # This function contains examples of calling your functions. However, you should use unitgrade to verify correctness.

    ## Problem 7: Lets try to go East. Run this code to see if the states you return looks sensible.
    states = go_east(east)
    for s in states:
        print(str(s))

    ## Problem 8: try the p_next function for a few empty environments. Does the result look sensible?
    x, _ = PacmanEnvironment(layout_str=east).reset()
    action = x.A()[0]
    print(f"Transitions when taking action {action} in map: 'east'")
    print(x)
    print(p_next(x, action))  # use str(state) to get a nicer representation.

    print(f"Transitions when taking action {action} in map: 'east2'")
    x, _ = PacmanEnvironment(layout_str=east2).reset()
    print(x)
    print(p_next(x, action))

    ## Problem 9
    print(f"Checking states space S_1 for k=1 in SS0tiny:")
    x, _ = PacmanEnvironment(layout_str=SS0tiny).reset()
    states = get_future_states(x, N=10)
    for s in states[1]: # Print all elements in S_1.
        print(s)
    print("States at time k=10, |S_10| =", len(states[10]))

    ## Problem 10
    N = 20  # Planning horizon
    action, states = shortest_path(east, N)
    print("east: Optimal action sequence:", action)

    action, states = shortest_path(datadiscs, N)
    print("datadiscs: Optimal action sequence:", action)

    action, states = shortest_path(SS0tiny, N)
    print("SS0tiny: Optimal action sequence:", action)


def one_ghost():
    # Win probability when planning using a single ghost. Notice this tends to increase with planning depth
    wp = []
    for n in range(10):
        wp.append(win_probability(SS1tiny, N=n))
    print(wp)
    print("One ghost:", win_probability(SS1tiny, N=12))


def two_ghosts():
    # Win probability when planning using two ghosts
    print("Two ghosts:", win_probability(SS2tiny, N=12))

if __name__ == "__main__":
    no_ghosts()
    one_ghost()
    two_ghosts()
