# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex05direct.direct_cartpole_kelly import compute_solutions
from irlc.ex05direct.direct_plot import plot_solutions
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env, solutions = compute_solutions()
    print("Did we succeed?", solutions[-1]['solver']['success'])
    plot_solutions(env, solutions, animate=True, pdf=None, animate_all=True, animate_repeats=3)
    env.close()
