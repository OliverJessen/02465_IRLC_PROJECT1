# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.lec11.mountain_car_env import FancyMountainCar
from irlc.pacman.pacman_resources import WHITE, BLACK
from irlc.utils.graphics_util_pygame import GraphicsUtilGym
from irlc.lectures.lec11.mountain_car_env import MountainCarVisualization
from irlc.ex11.semi_grad_sarsa import LinearSemiGradSarsa

if __name__ == '__main__':
    from irlc import Agent, interactive, train
    env = FancyMountainCar(render_mode='human')
    num_of_tilings = 8
    alpha = 0.3

    # env = gym.make("MountainCar-v0")
    agent = LinearSemiGradSarsa(env, gamma=1, alpha=alpha/num_of_tilings, epsilon=0)
    # agent = Agent(env)

    env, agent = interactive(env, agent)
    train(env, agent, num_episodes=10)

    env.close()



    pass
