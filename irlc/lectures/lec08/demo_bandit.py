# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.utils.bandit_graphics_environment import GraphicalBandit
import time
from irlc import train
from irlc.ex07.simple_agents import BasicAgent
from irlc import interactive

def bandit_eps(autoplay=False):
    env = GraphicalBandit(10, render_mode='human',frames_per_second=30)
    env.reset()
    agent = BasicAgent(env, epsilon=0.1)
    agent.method = 'Epsilon-greedy'
    env, agent = interactive(env, agent, autoplay=autoplay)

    t0 = time.time()
    n = 3000
    stats, _ = train(env, agent, max_steps=n, num_episodes=10, return_trajectory=False, verbose=False)
    tpf = (time.time()-t0)/ n
    print("tpf", tpf, 'fps', 1/tpf)
    env.close()

if __name__ == "__main__":
    bandit_eps()
