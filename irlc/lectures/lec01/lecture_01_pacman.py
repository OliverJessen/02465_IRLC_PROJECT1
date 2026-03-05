# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.pacman.pacman_environment import PacmanEnvironment
from irlc.ex01.agent import train, Agent
from irlc import interactive

def ppacman():
    # smallGrid
    env = PacmanEnvironment(layout='mediumClassic', render_mode='human')
    env, agent = interactive(env, Agent(env))

    stats, _ = train(env, agent, num_episodes=100, verbose=False)
    print("Accumulated reward", stats[-1]['Accumulated Reward'])
    env.close()

if __name__ == "__main__":
    ppacman()
