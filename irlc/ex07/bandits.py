# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (Freely available online).
"""
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import Env
from gymnasium.spaces import Discrete
from irlc import train
from tqdm import tqdm
import sys
from irlc import cache_read, cache_write, cache_exists

class BanditEnvironment(Env): 
    r"""
    A helper class for defining bandit problems similar to e.g. the 10-armed testbed discsused in (SB18).
    We are going to implement the bandit problems as greatly simplfied gym environments, as this will allow us to
    implement the bandit agents as the familiar ``Agent``. I hope this way of doing it will make it clearer that bandits
    are in fact a sort of reinforcement learning method.

    The following code shows an example of how to use a bandit environment:

    .. runblock:: pycon

        >>> from irlc.ex08.bandits import StationaryBandit
        >>> env = StationaryBandit(k=10)                    # 10-armed testbed.
        >>> env.reset()                                     # Reset env.q_star
        >>> s, r, _, _, info = env.step(3)
        >>> print(f"The reward we got from taking arm a=3 was {r=}")

    """
    def __init__(self, k : int): 
        r"""
        Initialize a bandit problem. The observation space is given a dummy value since bandit problems of the sort
        (SB18) discuss don't have observations.

        :param k: The number of arms.
        """
        super().__init__() 
        self.observation_space = Discrete(1)  # Dummy observation space with a single observation.
        self.action_space = Discrete(k)       # The arms labelled 0,1,...,k-1.
        self.k = k  # Number of arms 

    def reset(self): 
        r"""
        Use this function to reset the all internal parameters of the environment and get ready for a new episode.
        In the (SB18) 10-armed bandit testbed, this would involve resetting the expected return

        .. math::
            q^*_a

        The function must return a dummy state and info dictionary to agree with the gym ``Env`` class, but their values are
        irrelevant

        :return:
            - s - a state, for instance 0
            - info - the info dictionary, for instance {}
        """
        raise NotImplementedError("Implement the reset method") 

    def bandit_step(self, a): 
        r"""This helper function simplify the definition of the environments ``step``-function.

        Given an action :math:`r`, this function computes the reward obtained by taking that action :math:`r_t`
        and the gab. This is defined as the expected reward we miss out on by taking the potentially suboptimal action :math:`a`
        and is defined as:

        .. math::
            \Delta = \max_{a'} q^*_{a'} - q_a

        Once implemented, the reward and regret enters into the ``step`` function as follows:

        .. runblock:: pycon

            >>> from irlc.ex08.bandits import StationaryBandit
            >>> env = StationaryBandit(k=4)     # 4-armed testbed.
            >>> env.reset()                     # Reset all parameters.
            >>> _, r, _, _, info = env.step(2)  # Take action a=2
            >>> print(f"Reward from a=2 was {r=}, the gab was {info['gab']=}")

        :param a: The current action we take
        :return:
            - r - The reward we thereby incur
            - gab - The regret gab :math:`\Delta` incurred by taking this action (0 for an optimal action)
        """
        reward = 0 # Compute the reward associated with arm a 
        gab = 0 # Compute the gab, by comparing to the optimal arms reward.
        return reward, gab

    def step(self, action): 
        r"""You do not have to edit this function.
        In a bandit environment, the step function is simplified greatly since there are no
        states to keep track on. It should simply return the reward incurred by the action ``a``
        and (for convenience) also returns the gab in the ``info``-dictionary.

        :param action: The current action we take :math:`a_t`
        :return:
            - next_state - This is always ``None``
            - reward - The reward obtained by taking the given action. In (SB18) this is defined as :math:`r_t`
            - terminated - Always ``False``. Bandit problems don't terminate.
            - truncated - Always ``False``
            - info - For convenience, this includes the gab (used by the plotting methods)

        """
        reward, gab = self.bandit_step(action) 
        info = {'gab': gab}
        return None, reward, False, False, info  

class StationaryBandit(BanditEnvironment): 
    r"""Implement the 'stationary bandit environment' which is described in (SB18, Section 2.3)
    and used as a running example throughout the chapter.

    We will implement a version with a constant mean offset (q_star_mean), so that

     q* = x + q_star_mean,   x ~ Normal(0,1)

    q_star_mean can just be considered to be zero at first.
    """
    def __init__(self, k, q_star_mean=0):
        super().__init__(k)
        self.q_star_mean = q_star_mean

    def reset(self): 
        """ Set q^*_k = N(0,1) + mean_value. The mean_value is 0 in most examples. I.e., implement the 10-armed testbed environment. """
        self.q_star = np.random.randn(self.k) + self.q_star_mean
        self.optimal_action = np.argmax(self.q_star) # Optimal action is the one with the largest q^*-value. 
        return 0, {} # The reset method in a gym Env must return a (dummy) state and a dictionary.

    def bandit_step(self, a): 
        """ Return the reward/gab for action a for the simple bandit. Use self.q_star (see reset-function above).
         To implement it, implement the reward (see the description of the 10-armed testbed for more information.
         How is it computed from q^*_k?) and also compute the gab.

         As a small hint, since we are computing the gab, it will in fact be the difference between the
         value of q^* corresponding to the current arm, and the q^* value for the optimal arm.
         Remember it is 0 if the optimal action is selected.
         """
        # TODO: 2 lines missing.
        raise NotImplementedError("Insert your solution and remove this error.")
        # Actual logic goes here. Use self.q_star[a] to get mean reward and np.random.randn() to generate random numbers.  
        return reward, gab 

    def __str__(self):
        return f"{type(self).__name__}_{self.q_star_mean}"

"""
Helper function for running a bunch of bandit experiments and plotting the results.

The function will run the agents in 'agents' (a list of bandit agents) 
on the bandit environment 'bandit' and plot the result.

Each agent will be evaluated for num_episodes episodes, and one episode consist of 'steps' steps.
However, to speed things up you can use cache, and the bandit will not be evaluated for more than 
'max_episodes' over all cache runs. 

"""
def eval_and_plot(bandit, agents, num_episodes=2000, max_episodes=2000, steps=1000, labels=None, use_cache=True):
    if labels is None:
        labels = [str(agent) for agent in agents]

    f, axs = plt.subplots(nrows=3, ncols=1)
    f.set_size_inches(10,7)
    (ax1, ax2, ax3) = axs
    for i,agent in enumerate(agents):
        rw, oa, regret, num_episodes = run_agent(bandit, agent, episodes=num_episodes, max_episodes=max_episodes, steps=steps, use_cache=use_cache)
        ax1.plot(rw, label=labels[i])
        ax2.plot(oa, label=labels[i])
        ax3.plot(regret, label=labels[i])

    for ax in axs:
        ax.grid()
        ax.set_xlabel("Steps")

    ax1.set_ylabel("Average Reward")
    ax2.set_ylabel("% optimal action")
    ax3.set_ylabel("Regret $L_t$")
    ax3.legend()
    f.suptitle(f"Evaluated on {str(bandit)} for {num_episodes} episodes")

def run_agent(env, agent, episodes=2000, max_episodes=2000, steps=1000, use_cache=False, verbose=True):
    """
    Helper function. most of the work involves the cache; the actual training is done by 'train'.
    """
    C_regrets_cum_sum, C_oas_sum, C_rewards_sum, C_n_episodes = 0, 0, 0, 0
    if use_cache:
        cache = f"cache/{str(env)}_{str(agent)}_{steps}.pkl"
        if cache_exists(cache):
            print("> Reading from cache", cache)
            C_regrets_cum_sum, C_oas_sum, C_rewards_sum, C_n_episodes = cache_read(cache)

    regrets = []
    rewards = []
    cruns = max(0, min(episodes, max_episodes - C_n_episodes)) # Missing runs.
    for _ in tqdm(range(cruns), file=sys.stdout, desc=str(agent),disable=not verbose):
        stats, traj = train(env, agent, max_steps=steps, verbose=False, return_trajectory=True)
        regret = np.asarray([r['gab'] for r in traj[0].env_info[1:]])
        regrets.append(regret)
        rewards.append(traj[0].reward)

    regrets_cum_sum = C_regrets_cum_sum
    oas_sum = C_oas_sum
    rewards_sum = C_rewards_sum
    episodes = C_n_episodes
    if len(regrets) > 0:
        regrets_cum_sum += np.cumsum(np.sum(np.stack(regrets), axis=0))
        oas_sum += np.sum(np.stack(regrets) == 0, axis=0)
        rewards_sum += np.sum(np.stack(rewards), axis=0)
        episodes += cruns
    if use_cache and cruns > 0:
        cache_write((regrets_cum_sum, oas_sum, rewards_sum, episodes), cache, protocol=4)
    return rewards_sum/episodes, oas_sum/episodes, regrets_cum_sum/episodes, episodes
