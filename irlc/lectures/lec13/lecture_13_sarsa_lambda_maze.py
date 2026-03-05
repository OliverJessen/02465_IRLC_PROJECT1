# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.lec13.lecture_13_Q_maze import sutton_maze_play
from irlc.ex12.sarsa_lambda_agent import SarsaLambdaAgent

if __name__ == "__main__":
    sutton_maze_play(SarsaLambdaAgent, method_label="Sarsa(Lambda=0.9)", lamb=0.9)
