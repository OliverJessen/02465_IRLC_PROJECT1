# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex01.agent import train, Agent
from irlc.ex04.model_pendulum import GymSinCosPendulumEnvironment

if __name__ == "__main__":
    env = GymSinCosPendulumEnvironment(Tmax=100, render_mode='human')
    agent = Agent(env)
    stats, _ = train(env, agent, num_episodes=1, verbose=False)
    env.close()
