# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc import train, interactive
from irlc.lectures.lec01.viz_inventory_environment import VizInventoryEnvironment

if __name__ == "__main__":
    env = VizInventoryEnvironment(render_mode='human')
    from irlc.ex02.inventory import InventoryDPModel
    from irlc.ex02.dp_agent import DynamicalProgrammingAgent
    agent = DynamicalProgrammingAgent(env, model=InventoryDPModel())

    env, agent = interactive(env, agent)
    n = 400
    stats, _ = train(env, agent, max_steps=n, num_episodes=1000, return_trajectory=False, verbose=False)
