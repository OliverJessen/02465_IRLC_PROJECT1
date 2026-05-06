# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.lec10.lecture_10_sarsa_open import open_play
from irlc.lectures.lec12.sarsa_lambda_delay import SarsaLambdaDelayAgent

if __name__ == "__main__":
    open_play(SarsaLambdaDelayAgent, method_label="Sarsa(Lambda)", lamb=0.8)
