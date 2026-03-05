# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from irlc.lectures.lec11.mountain_car_env import FancyMountainCar
from irlc.ex11.semi_grad_sarsa import LinearSemiGradSarsa

class RandomWeightAgent(LinearSemiGradSarsa):
    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.Q.w = np.random.randn(self.Q.w.shape[0])

if __name__ == '__main__':
    from irlc import Agent, interactive, train
    env = FancyMountainCar(render_mode='human')
    num_of_tilings = 8
    alpha = 0.3
    # env = gym.make("MountainCar-v0")
    agent = RandomWeightAgent(env) #(env, gamma=1, alpha=alpha/num_of_tilings, epsilon=0)
    env, agent = interactive(env, agent)
    train(env, agent, num_episodes=10)

    env.close()



    pass
