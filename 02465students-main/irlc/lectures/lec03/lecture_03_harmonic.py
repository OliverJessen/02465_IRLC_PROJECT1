# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc import train
from irlc.ex04.model_harmonic import HarmonicOscilatorEnvironment
from irlc import Agent
import numpy as np

class NullAgent(Agent):
    def pi(self, x, k, info=None):
        return np.asarray([0])

if __name__ == "__main__":
    env = HarmonicOscilatorEnvironment(render_mode='human')
    train(env, NullAgent(env), num_episodes=1, max_steps=1000)
    env.close()
