# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from collections import defaultdict
import gymnasium as gym
from irlc.ex01.agent import train
from irlc import main_plot, savepdf
import matplotlib.pyplot as plt
from irlc.ex10.sarsa_agent import SarsaAgent


class SarsaLambdaAgent(SarsaAgent):
    def __init__(self, env, gamma=0.99, epsilon=0.1, alpha=0.5, lamb=0.9):
        """
        Implementation of Sarsa(Lambda) in the tabular version, see
        http://incompleteideas.net/book/first/ebook/node77.html
        for details. Remember to reset the
        eligibility trace E after each episode, i.e. set E(s,a) = 0.

        Note 'lamb' is an abbreveation of lambda, because lambda is a reserved keyword in python.

        The constructor initializes e, the eligibility trace. Since we want to easily be able to find the non-zero
        elements it will be convenient to use a dictionary. I.e.

        self.e[(s,a)] is the eligibility trace e(s,a) (or E(s,a) if you prefer).

        Note that Sarsa(Lambda) generalize Sarsa. This means that we again must generate the next action A' from S' in the train method and
        store it for when we take actions in the policy method pi. I.e. we can re-use the Sarsa Agents code for the policy (self.pi).
        """
        super().__init__(env, gamma=gamma, alpha=alpha, epsilon=epsilon)
        self.lamb = lamb
        # We use a dictionary to store the eligibility trace. It can be indexed as self.e[s,a].
        self.e = defaultdict(float)

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        # TODO: 1 lines missing.
        raise NotImplementedError("a_prime = ... (get action for S'=sp using self.pi_eps; see Sarsa)")
        # TODO: 1 lines missing.
        raise NotImplementedError("delta = ... (The ordinary Sarsa learning signal)")
        # TODO: 1 lines missing.
        raise NotImplementedError("Update the eligibility trace e(s,a) += 1")
        for (s,a), ee in self.e.items():
            # TODO: 2 lines missing.
            raise NotImplementedError("Update Q values and eligibility trace")
        if done: # Clear eligibility trace after each episode and update variables for Sarsa
            self.e.clear()
        else:
            self.a = a_prime

    def __str__(self):
        return f"SarsaLambda_{self.gamma}_{self.epsilon}_{self.alpha}_{self.lamb}"

if __name__ == "__main__":
    envn = 'CliffWalking-v0'
    env = gym.make(envn)

    alpha =0.05
    sarsaLagent = SarsaLambdaAgent(env,gamma=0.99, epsilon=0.1, alpha=alpha, lamb=0.9)
    sarsa = SarsaAgent(env,gamma=0.99,alpha=alpha,epsilon=0.1)
    methods = [("SarsaL", sarsaLagent), ("Sarsa", sarsa)]

    experiments = []
    for k, (name,agent) in enumerate(methods):
        expn = f"experiments/{envn}_{name}"
        train(env, agent, expn, num_episodes=500, max_runs=10)
        experiments.append(expn)
    main_plot(experiments, smoothing_window=10, resample_ticks=200)
    plt.ylim([-100, 0])
    savepdf("cliff_sarsa_lambda")
    plt.show()
