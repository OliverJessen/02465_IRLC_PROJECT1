# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from collections import defaultdict
import matplotlib.pyplot as plt
from irlc.ex08.rl_agent import TabularAgent
from irlc import main_plot, savepdf, train
from irlc import interactive


def get_MC_return_SA(episode, gamma, first_visit=True):
    """ Helper method for computing the MC returns.
    Given an episodes in the form [ (s0,a0,r1), (s1,a1,r2), ...]
    this function computes (if first_visit=True) a new list

    > [((s,a), G) , ... ]

    consisting of the unique $(s_t,a_t)$ pairs in episode along with their return G_t (computed from their first occurance).
    Alternatively, if first_visit=False, the method return a list of same length of episode
    with all (s,a) pairs and their return.
    """
    sa = [(s, a) for s, a, r in episode] # Get all state/action pairs. Useful for checking if we have visited a state/action before.
    G = 0
    returns = []
    for t in reversed(range(len(episode))):
        G = gamma * G + episode[t][2] 
        sa_t = episode[t][:2] 
        if sa_t not in sa[:t] or not first_visit: 
            returns.append(sa_t + (G,) )
    return returns

class MCAgent(TabularAgent): 
    def __init__(self, env, gamma=1.0, epsilon=0.05, alpha=None, first_visit=True):
        if alpha is None:
            self.returns_sum_S = defaultdict(float)
            self.returns_count_N = defaultdict(float)
        self.alpha = alpha
        self.first_visit = first_visit
        self.episode = []
        super().__init__(env, gamma, epsilon) 

    def pi(self, s,k, info=None): 
        """
        Compute the policy of the MC agent. Remember the agent is epsilon-greedy. You can use the pi_eps(s,info)-function defined
        in the TabularAgent class.
        """
        return self.pi_eps(s, info)

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):  
        """
        Consult your implementation of value estimation agent for ideas. Note you can index the Q-values as

        >> self.Q[s, a] = new_q_value

        see comments in the Agent class for more details, however for now you can consider them as simply a nested
        structure where ``self.Q[s, a]`` defaults to 0 unless the Q-value has been updated.
        """
        self.episode.append((s, a, r))
        if done:
            returns = get_MC_return_SA(self.episode, self.gamma, self.first_visit)
            for s, a, G in returns:
                # s,a = sa
                if self.alpha is None:
                    self.returns_sum_S[s, a] += G
                    self.returns_count_N[s, a] += 1
                    self.Q[s, a] = self.returns_sum_S[s, a] / self.returns_count_N[s, a]
                else:
                    self.Q[s, a] += self.alpha * (G - self.Q[s, a])
            self.episode = [] 

    def __str__(self):
        return f"MC_{self.gamma}_{self.epsilon}_{self.alpha}_{self.first_visit}"

if __name__ == "__main__":
    """ Load environment but make sure it is time-limited. Can you tell why? """
    envn = "SmallGridworld-v0"

    from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment, BookGridEnvironment
    env = SuttonCornerGridEnvironment(uniform_initial_state=True)
    # env = BookGridEnvironment(living_reward=-0.05) # Uncomment to test an alternative environment with a negative living reward.

    gamma = 1 
    episodes = 20000
    experiment="experiments/mcagent_smallgrid"
    agent = MCAgent(env, gamma=gamma, first_visit=True)
    train(env, agent, experiment_name=experiment, num_episodes=episodes, return_trajectory=False)
    main_plot(experiments=[experiment], resample_ticks=200) 
    plt.title("Smallgrid MC agent value function")
    plt.ylim([-10, 0])
    savepdf("mcagent_smallgrid") 
    plt.show() 

    env, agent = interactive(env, agent)
    env.reset()
    env.plot()
    plt.title(f"MC on-policy control of {envn} using first-visit")
    savepdf("MC_agent_value_smallgrid")
    plt.show(block=False)
