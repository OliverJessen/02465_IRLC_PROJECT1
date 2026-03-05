# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.pacman.pacman_environment import PacmanEnvironment, PacmanWinWrapper
# from irlc.berkley.rl.feature_encoder import SimplePacmanExtractor
# from irlc.utils.player_wrapper_pyglet import PlayWrapper
import matplotlib.pyplot as plt
# from irlc.utils.video_monitor import VideoMonitor
from irlc.ex01.agent import train
# from irlc.lectures.lecture_09_mc import keyboard_play
from irlc.ex10.q_agent import QAgent
from irlc import interactive


def play_pacman(env, agent, layout = 'smallGrid'):

    train(env, agent, num_episodes=100)
    env2 = PacmanWinWrapper(env)
    # env2 = Monitor(env2, directory="experiments/randomdir", force=True)
    # env2 = VideoMonitor(env2)
    env2, agent = interactive(env2, agent)
    agent.epsilon = 0
    agent.alpha = 0
    # agent = PlayWrapper(agent, env2)
    train(env2, agent, num_episodes=100)
    plt.show()
    env.close()

if __name__ == "__main__":
    layout = 'smallGrid'
    env = PacmanEnvironment(animate_movement=False, layout=layout, render_mode='human')
    agent = QAgent(env, epsilon=0.05, alpha=0.1, gamma=0.8)
    # from irlc import PlayWrapper
    # agent = PlayWrapper(agent, env)
    play_pacman(env, agent, layout = 'smallGrid')
    # main_plot('experiments/q_lin')
