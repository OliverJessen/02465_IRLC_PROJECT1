# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.pacman.pacman_environment import PacmanEnvironment
from irlc.ex02.dp_agent import DynamicalProgrammingAgent
from gymnasium.wrappers import TimeLimit
from irlc.pacman.pacman_environment import PacmanWinWrapper
from irlc.ex01.agent import train
from irlc.lectures.chapter3dp.dp_pacman import DPPacmanModel
from irlc import interactive

def simulate_1_game(layout_str):
    N = 30
    env = PacmanEnvironment(layout=None, layout_str=layout_str, render_mode='human')
    model = DPPacmanModel(env, N=N, verbose=True)
    agent = DynamicalProgrammingAgent(env, model=model)
    env, agent = interactive(env, agent)
    env = TimeLimit(env, max_episode_steps=N)
    env = PacmanWinWrapper(env)
    stats, trajectories = train(env, agent, num_episodes=100, verbose=False, return_trajectory=True)
    env.close()


SS0 = """
%%%%%%%%%%
% P  .   %
% %%%%%. %
%        %
% %%% %%%%
%.      .%
%%%%%%%%%%
"""
if __name__ == "__main__":
    simulate_1_game(layout_str=SS0)
