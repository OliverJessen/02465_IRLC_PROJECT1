# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.pacman.pacman_environment import PacmanEnvironment
from irlc.ex01.agent import train
from irlc.ex01.agent import Agent
from irlc import interactive
from irlc.lectures.chapter3dp.dp_pacman import SS1tiny


def ppac(layout_str, name="pac"):
    env = PacmanEnvironment(layout=None, layout_str=layout_str, animate_movement=True)
    agent = Agent(env)
    env, agent = interactive(env, agent)

    stats, _ = train(env, agent, num_episodes=5, max_steps=8)
    print("Accumulated reward for all episodes:", [s['Accumulated Reward'] for s in stats])
    env.close()

if __name__ == "__main__":
    ppac(SS1tiny)
