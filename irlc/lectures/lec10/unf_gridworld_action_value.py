# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex01.agent import Agent
from irlc.gridworld.gridworld_environments import BookGridEnvironment, FrozenLakeEnv
# from irlc.utils.video_monitor import VideoMonitor
from irlc import interactive, train
# from irlc.ex01.agent import train
# from irlc import PlayWrapper

from irlc.ex09.mc_agent import MCAgent

class SingleActionValueAgent(MCAgent):
    def __init__(self, env, gamma=1.0, epsilon=0.05, alpha=None, first_visit=True):
        super().__init__(env, gamma=1., epsilon=1, alpha=None, first_visit=True)

    def pi(self, s, k, info=None):
        if k == 0:
            return 1
        else:
            return super().pi_eps(s, info=None)

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        super().train(s, a, r, sp, done, info_s, info_sp)
        for s in self.env.mdp.nonterminal_states:
            for a in self.env.mdp.A(s):
                if s == (0,0) and a == 1:
                    pass
                elif len(self.env.mdp.A(s)) == 1:
                    pass
                else:
                    self.Q[s,a] = 0
        a = 234





if __name__ == "__main__":
    env = BookGridEnvironment(render_mode='human', print_states=True, living_reward=-0.05)
    env, agent = interactive(env, SingleActionValueAgent(env))
    agent.label = "Random agent"
    train(env, agent, num_episodes=100, verbose=False)
    env.close()
