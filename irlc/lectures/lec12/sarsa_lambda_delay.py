# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from collections import defaultdict
from irlc.ex10.q_agent import QAgent

class SarsaLambdaDelayAgent(QAgent):
    def __init__(self, env, gamma=0.99, epsilon=0.1, alpha=0.5, lamb=0.9):
        super().__init__(env, gamma=gamma, alpha=alpha, epsilon=epsilon)
        self.lamb = lamb
        self.method = 'Sarsa(Lambda)'
        self.e = defaultdict(float)

    def pi(self, s, k, info=None):
        self.t = k
        action = self.pi_eps(s,info=info)
        return action

    def lmb_update(self, s, a, r, sp, ap, done):
        delta = r + self.gamma * (self.Q[sp,ap] if not done else 0) - self.Q[s,a]
        for (s,a), ee in self.e.items():
            self.Q[s,a] += self.alpha * delta * ee
            self.e[(s,a)] = self.gamma * self.lamb * ee

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        # if self.t == 0:
        #     self.e.clear()

        if self.t > 0:
            # We have an update in the buffer and can update the states.
            self.lmb_update(self.s_prev, self.a_prev, self.r_prev, s, a, done=False)
        self.e[(s, a)] += 1

        if done:
            self.lmb_update(s, a, r, sp, ap=None, done=True)
            self.e.clear()

        self.s_prev = s
        self.a_prev = a
        self.r_prev = r

    def __str__(self):
        return f"SarsaLambdaDelay_{self.gamma}_{self.epsilon}_{self.alpha}_{self.lamb}"

if __name__ == "__main__":
    from irlc.ex12.sarsa_lambda_open import keyboard_play
    keyboard_play(SarsaLambdaDelayAgent, method_label="Sarsa(Lambda) (delayed)")
