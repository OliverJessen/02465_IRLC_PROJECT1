# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
if __name__ == "__main__":
    from irlc.lectures.lec06.lecture_06_pendulum_bilqr_ubar import pen_experiment
    N = 50
    pen_experiment(N=N, use_linesearch=True, use_ubar=False)
