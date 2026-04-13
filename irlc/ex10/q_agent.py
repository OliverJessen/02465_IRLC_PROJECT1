# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (Freely available online).
"""
from irlc.ex08.mdp import GymEnv2MDP
from irlc.ex08.rl_agent import TabularAgent
from irlc import train
import gymnasium as gym
from irlc import main_plot
import matplotlib.pyplot as plt
from irlc import savepdf
from irlc.ex08.value_iteration_agent import ValueIterationAgent

class QAgent(TabularAgent):
    r"""
    Implement the Q-learning agent (SB18, Section 6.5)
    Note that the Q-datastructure already exist, as do helper functions useful to compute an epsilon-greedy policy.
    You can access these as

    > self.Q[s,a] = 31 # Set a Q-value.

    See the TabularAgent class for more information.
    """
    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1):
        self.alpha = alpha
        super().__init__(env, gamma, epsilon)

    def pi(self, s, k, info=None): 
        """
        Return current action using epsilon-greedy exploration. You should look at the TabularAgent class for ideas.
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement the epsilon-greedy policy here.")
        return action

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None): 
        """
        Implement the Q-learning update rule, i.e. compute a* from the Q-values.
        As a hint, note that self.Q[sp,a] corresponds to q(s_{t+1}, a) and
        that what you need to update is self.Q[s, a] = ...

        You may want to look at self.Q.get_optimal_action(state) to compute a = argmax_a Q[s,a].
        """
        # TODO: 3 lines missing.
        raise NotImplementedError("Update the Q[s,a]-values here.")

    def __str__(self):
        return f"QLearner_{self.gamma}_{self.epsilon}_{self.alpha}"

q_exp = f"experiments/cliffwalk_Q"
epsilon = 0.1
max_runs = 10
alpha = 0.1
def cliffwalk():
    env = gym.make('CliffWalking-v1')
    agent = QAgent(env, epsilon=epsilon, alpha=alpha)
    train(env, agent, q_exp, num_episodes=200, max_runs=max_runs)

    # As a baseline, we set up/evaluate a value-iteration agent to get an idea about the optimal performance.
    # To do so, we need an MDP object. We create an MDP object out of the gym environment below.
    # You can look at the code if you like, but it is simply a helper function to convert from one datastructure to another,
    # and all it does is to give a MDP object which is needed for our value-iteration implementation from the previous
    # week.
    mdp = GymEnv2MDP(env)
    vi_exp = "experiments/cliffwalk_VI"
    Vagent = ValueIterationAgent(env, mdp=mdp, epsilon=epsilon)
    train(env, Vagent, vi_exp, num_episodes=200, max_runs=max_runs)

    vi_exp_opt = "experiments/cliffwalk_VI_optimal"
    Vagent_opt = ValueIterationAgent(env, mdp=mdp, epsilon=0) # Same, but with epsilon=0
    train(env, Vagent_opt, vi_exp_opt, num_episodes=200, max_runs=max_runs)

    exp_names = [q_exp, vi_exp, vi_exp_opt]
    return env, exp_names

if __name__ == "__main__":
    for _ in range(10):
        env, exp_names = cliffwalk()
    main_plot(exp_names, smoothing_window=10)
    plt.ylim([-100, 0])
    plt.title("Q-learning on " + env.spec.name)
    savepdf("Q_learning_cliff")
    plt.show()
