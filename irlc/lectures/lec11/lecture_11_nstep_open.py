# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.lec10.sarsa_nstep_delay import SarsaDelayNAgent
from irlc.lectures.lec11.lecture_10_sarsa_open import open_play

if __name__ == "__main__":
    # env = OpenGridEnvironment()
    # agent = (env, gamma=0.95, epsilon=0.1, alpha=.5)
    open_play(SarsaDelayNAgent, method_label="N-step Sarsa n=8", n=8)
