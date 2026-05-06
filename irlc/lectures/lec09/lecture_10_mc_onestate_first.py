# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.exam_tabular_examples.helper import keyboard_play_value
from irlc.ex09.mc_evaluate import MCEvaluationAgent
from irlc.gridworld.gridworld_environments import GridworldEnvironment

map = [['#', '#', '#', '#'],
        ['#','S',0,'#'],
        ['#','#','#','#']]

class CaughtGrid(GridworldEnvironment):
    def __init__(self, **kwargs):
        super().__init__(map, living_reward=1, zoom=1.5, **kwargs)



if __name__ == "__main__":
    env = CaughtGrid(view_mode=1, render_mode='human')
    agent = MCEvaluationAgent(env, gamma=1, alpha=None)
    keyboard_play_value(env,agent,method_label='MC (first visit)')
