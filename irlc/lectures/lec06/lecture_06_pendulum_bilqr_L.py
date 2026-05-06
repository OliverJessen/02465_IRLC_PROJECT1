# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from irlc.lectures.lec06.lecture_06_pendulum_bilqr_ubar import pen_experiment

good_seed = 2
bad_seed = 1

if __name__ == "__main__":
    #np.random.seed(good_seed) # (2: ok, 1: fail).
    np.random.seed(bad_seed) # (2: ok, 1: fail).
    pen_experiment(N=50, use_linesearch=False, use_ubar=False)
