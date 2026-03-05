# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc import train
from irlc.gridworld.gridworld_environments import OpenGridEnvironment
from irlc.lectures.lec11.lecture_10_sarsa_open import open_play
from irlc.lectures.lec11.sarsa_nstep_delay import SarsaDelayNAgent

if __name__ == "__main__":
    env = OpenGridEnvironment()
    agent = SarsaDelayNAgent(env, n=1)
    train(env, agent, num_episodes=100)
    open_play(SarsaDelayNAgent, method_label=f"Sarsa")
