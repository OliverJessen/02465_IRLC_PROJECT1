# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex13.dyna_q import DynaQ
from irlc.lectures.lec10.lecture_10_mc_q_estimation import keyboard_play
from irlc.gridworld.gridworld_environments import SuttonMazeEnvironment

def sutton_maze_play(Agent, method_label="Q-learning agent", **kwargs):
    env = SuttonMazeEnvironment(render_mode="human")
    agent = Agent(env, gamma=0.98, epsilon=0.1, alpha=.5, **kwargs)
    keyboard_play(env, agent, method_label=method_label)

if __name__ == "__main__":
    sutton_maze_play(DynaQ, method_label="Q-learning agent", n=0)
