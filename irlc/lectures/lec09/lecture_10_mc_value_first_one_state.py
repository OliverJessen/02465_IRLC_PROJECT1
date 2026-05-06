# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.lec09.lecture_10_mc_q_estimation import keyboard_play
from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment, BookGridEnvironment
from irlc.ex09.mc_agent import MCAgent
from irlc.ex09.mc_evaluate import MCEvaluationAgent
import numpy as np
from irlc import interactive, train


class MCAgentOneState(MCEvaluationAgent):
    def __init__(self, *args, state=None, **kwargs):
        a = 34
        super().__init__(*args, **kwargs)
        if state is None:
            state = self.env.mdp.initial_state
        self.state = state
        self._clear_states()

    def _clear_states(self, val=None):
        for s in self.env.mdp.nonterminal_states:
            if s != self.state:
                self.returns_sum_S[s] = val
                self.returns_count_N[s] = val
                if s in self.v:
                    k = next(self.env.mdp.Psr(s, self.env.mdp.A(s)[0]).keys().__iter__() )[0]
                    if not self.env.mdp.is_terminal(k):

                        del self.v[s]

    def reset(self):
        from irlc.lectures.lec10.utils import agent_reset
        agent_reset(self)
        self._clear_states(None)

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        # self.episode = [e for e in self.episode if e[0] == self.state]
        self._clear_states(0)
        super().train(s, a, r, sp, done)
        self._clear_states(None)


if __name__ == "__main__":
    env = BookGridEnvironment(render_mode='human', living_reward=-0.05, print_states=True, zoom=2)

    agent = MCAgentOneState(env, gamma=1, alpha=None, first_visit=True)
    method_label = 'MC (gamma=1)'
    agent.label = method_label
    autoplay = False
    env, agent = interactive(env, agent, autoplay=autoplay)
    # agent = PlayWrapper(agent, env,autoplay=autoplay)
    # env = VideoMonitor(env, agent=agent, fps=100, agent_monitor_keys=('pi', 'Q'), render_kwargs={'method_label': method_label})
    num_episodes = 1000
    train(env, agent, num_episodes=num_episodes)
    env.close()
