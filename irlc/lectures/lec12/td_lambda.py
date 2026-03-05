# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex08.rl_agent import ValueAgent
from collections import defaultdict

class TDLambdaAgent(ValueAgent):
    def __init__(self, env, gamma=0.99, alpha=0.5, lamb=0.9):
        self.alpha = alpha
        self.lamb = lamb
        self.e = defaultdict(float)
        super().__init__(env, gamma=gamma)

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        for ss, ee in self.e.items():
            self.e[ss] = self.gamma * self.lamb * ee
            pass

        self.e[s] += 1
        delta = r + self.gamma * (self.v[sp] if not done else 0) - self.v[s]
        for ss, ee in self.e.items():
            self.v[ss] += self.alpha * delta * ee
            # self.e[ss] = self.gamma * self.lamb * ee

        if done:
            self.e.clear()

    def __str__(self):
        return f"TD(Lambda={self.lamb})_value_{self.gamma}_{self.alpha}"

if __name__ == "__main__":
    from irlc.lectures.lec10.lecture_10_mc_q_estimation import keyboard_play
    from irlc.gridworld.gridworld_environments import OpenGridEnvironment

    env = OpenGridEnvironment(render_mode='human', frames_per_second=30)
    agent = TDLambdaAgent(env, gamma=1, alpha=.5, lamb=0.9)
    method_label = 'TD(Lambda)'
    method_label = f"{method_label} (gamma=0.99, alpha=0.5)"
    keyboard_play(env, agent, method_label=method_label)
    env.close()
    # if __name__ == "__main__":
    #
    # open_play(TDLambdaAgent, method_label="TD(Lambda)", lamb=0.8)
    #
    # pass
