# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc import Agent, interactive
from irlc.lectures.lec01.viz_inventory_environment import VizInventoryEnvironment

class OneAgent(Agent):
    def pi(self, s, k, info):
        return 1

if __name__ == "__main__":
    env = VizInventoryEnvironment(render_mode='human')
    from irlc import train
    from irlc import Agent

    # from irlc.ex08.ucb_agent import UCBAgent
    # from irlc.utils.player_wrapper import PlayWrapper
    from irlc import interactive
    # agent = BasicAgent(env, epsilon=0.1)
    # agent = UCBAgent(env)
    #agent = Agent(env)
    agent = OneAgent(env)

    env, agent = interactive(env, agent)
    n = 400
    stats, _ = train(env, agent, max_steps=n, num_episodes=1000, return_trajectory=False, verbose=False)
