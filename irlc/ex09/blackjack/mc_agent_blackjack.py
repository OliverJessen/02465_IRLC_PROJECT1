# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from irlc import main_plot
from irlc import savepdf
from irlc.ex01.agent import train
from irlc.ex09.mc_evaluate_blackjack import plot_blackjack_value, plot_blackjack_policy
from irlc.ex09.mc_agent import MCAgent

def run_experiment(episodes, first_visit=True, **kwargs):
    env_name = 'Blackjack-v1'
    env = gym.make(env_name)
    agent = MCAgent(env, **kwargs)
    lbl = "_".join(map(str, kwargs.values()))
    fvl = "First" if first_visit else "Every"
    title = f"MC agent ({fvl} visit)"

    expn = f"experiments/{env_name}_MCagent_{episodes}_{first_visit}_{lbl}" # Name the experiment. Pass the label to the train function to store intermediate results. See the online documentation for more information.
    # TODO: 1 lines missing.
    raise NotImplementedError("call the train(...) function here.")

    # Matplotlib with seaborn is for some reason very slow.
    # This code re-samples the curve to just 400 points:
    main_plot(expn, smoothing_window=episodes//100, resample_ticks=400)
    plt.title("Estimated returns in blackjack using " + title)
    plt.ylim([-0.3, 0])
    savepdf(f"blackjack_MC_agent_{episodes}_{first_visit}")
    plt.show()

    V = defaultdict(lambda: 0)
    A = defaultdict(lambda: 0)
    for s, av in agent.Q.to_dict().items():
        A[s] = agent.pi(s, 0)
        V[s] = max(av.values() )

    plot_blackjack_value(V, title=title, pdf_out=f"blackjack_mcagent_policy{fvl}_valfun_{episodes}")
    plt.show()
    plot_blackjack_policy(A, title=title)
    savepdf(f"blackjack_mcagent_policy{fvl}_{episodes}")
    plt.show()

if __name__ == "__main__":
    episodes = 1000000
    # episodes = 1000 # Uncomment to run far fewer episodes during debugging.
    run_experiment(episodes, epsilon=0.05, first_visit=True)
    run_experiment(episodes, epsilon=0.05, first_visit=False)
