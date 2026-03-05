# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex13.dyna_q import DynaQ
from irlc.lectures.lec13.lecture_13_Q_maze import sutton_maze_play

if __name__ == "__main__":
    sutton_maze_play(DynaQ, method_label="DynaQ (n=5)", n=5)
