# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex01.agent import train
from irlc import main_plot
import matplotlib.pyplot as plt
from irlc import savepdf
from irlc.ex10.sarsa_agent import SarsaAgent
from irlc.ex10.q_agent import QAgent
from irlc.ex13.tabular_double_q import TabularDoubleQ
from irlc.ex08.rl_agent import TabularQ
from irlc.gridworld.gridworld_environments import CliffGridEnvironment

class DoubleQVizAgent(TabularDoubleQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = TabularQ(self.env)

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        super().train(s, a, r, sp, done, info_s,info_sp)
        self.Q[s,a] = (self.Q1[s,a] + self.Q2[s,a] )/2

def train_cliff(runs=4, extension="long", save_pdf=False, alpha=0.02, num_episodes=5000):
    """ Part 1: Cliffwalking """
    # env = gym.make('CliffWalking-v0')

    env = CliffGridEnvironment(zoom=1)
    epsilon = 0.1
    # alpha = 0.02
    for _ in range(runs):
        agents = [QAgent(env, gamma=1, epsilon=epsilon, alpha=alpha),
                  SarsaAgent(env, gamma=1, epsilon=epsilon, alpha=alpha),
                  DoubleQVizAgent(env, gamma=1, epsilon=epsilon, alpha=alpha)]

        experiments = []
        for agent in agents:
            expn = f"experiments/doubleq_cliffwalk_{extension}_{str(agent)}"
            train(env, agent, expn, num_episodes=num_episodes, max_runs=1e6)
            experiments.append(expn)
    if save_pdf:
        main_plot(experiments, smoothing_window=20, resample_ticks=500)
        plt.ylim([-100, 50])
        plt.title(f"Double-Q learning on Cliffwalk ({extension})")
        savepdf(f"double_Q_learning_cliff_{extension}")
        plt.show()
    return agents, env


def grid_experiment(runs=20, extension="long", alpha=0.02, num_episodes=5000):
    from irlc.gridworld.gridworld_environments import CliffGridEnvironment
    # from irlc import VideoMonitor, PlayWrapper
    from irlc import interactive

    agents, env = train_cliff(runs=runs, extension=extension, save_pdf=True, alpha=alpha, num_episodes=num_episodes)
    labels = ["Q-learning", "Sarsa", "Double Q-learning"]
    for na in range(len(agents)):
        env2 = CliffGridEnvironment(zoom=1, view_mode='human')
        env2, agent = interactive(env2, agent=agents[na])# , agent_monitor_keys=('Q',), render_kwargs={'method_label': labels[na]})
        # agent = PlayWrapper(agents[na], env)
        env2.savepdf(f"doubleq_cliff_{extension}_agent_{na}")
        env2.close()

    env.close()
    pass

if __name__ == "__main__":
    """ 
    Test cliffwalk in both the long and short version
    """
    grid_experiment(runs=1, extension="long", alpha=0.02, num_episodes=5000)
    grid_experiment(runs=1, extension="short", alpha=0.25, num_episodes=500)
