# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc import Agent, train
from irlc.ex06.model_cartpole import GymSinCosCartpoleEnvironment

if __name__ == "__main__":

    env = GymSinCosCartpoleEnvironment(Tmax=20, render_mode='human')
    train(env, Agent(env), num_episodes=1)
    env.close()
