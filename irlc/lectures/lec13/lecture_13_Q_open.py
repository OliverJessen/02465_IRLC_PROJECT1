# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.lec11.lecture_10_sarsa_open import open_play
from irlc.ex10.q_agent import QAgent

if __name__ == "__main__":
    open_play(QAgent, method_label="Q-learning agent")
