# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.car.car_model import CarEnvironment
from irlc.ex01.agent import train, Agent
from irlc import interactive

if __name__ == "__main__":
    env = CarEnvironment(render_mode='human')
    env.action_space.low[1] = 0  # To ensure we do not drive backwards.
    agent = Agent(env)
    env, agent = interactive(env, agent, autoplay=False)
    stats, _ = train(env, agent, num_episodes=10, verbose=False)
    env.close()
