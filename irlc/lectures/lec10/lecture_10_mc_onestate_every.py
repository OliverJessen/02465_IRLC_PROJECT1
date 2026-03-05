# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.exam_tabular_examples.helper import keyboard_play_value
from irlc.ex09.mc_evaluate import MCEvaluationAgent
from irlc.lectures.lec10.lecture_10_mc_onestate_first import CaughtGrid


if __name__ == "__main__":
    env = CaughtGrid(view_mode=1, render_mode='humanp')
    agent = MCEvaluationAgent(env, gamma=1, alpha=None, first_visit=False)
    keyboard_play_value(env,agent,method_label='MC (every visit)')
