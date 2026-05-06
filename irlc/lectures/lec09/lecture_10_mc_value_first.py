# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.gridworld.gridworld_environments import BookGridEnvironment, BridgeGridEnvironment, GridworldEnvironment
from irlc.ex09.mc_evaluate import MCEvaluationAgent
from irlc import interactive, train

class BridgeGridEnvironment2(GridworldEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_bridge_grid, *args, **kwargs)


grid_bridge_grid = [[ '#',-100, -100, -100, -100, -100, '#'],
        [   1, ' ',  'S',  ' ',  ' ',  ' ',  2],
        [ '#',-100, -100, -100, -100, -100, '#']]


if __name__ == "__main__":

    # env = BridgeGridEnvironment2(view_mode=1, render_mode='human', living_reward=0)
    # agent = MCEvaluationAgent(env, gamma=.8, alpha=None, first_visit=False)
    # env, agent = interactive(env, agent)
    # train(env, agent, num_episodes=1000)
    # env.close()

    env = BookGridEnvironment(view_mode=1, render_mode='human', living_reward=-0.05)
    agent = MCEvaluationAgent(env, gamma=1, alpha=None)
    # agent = PlayWrapper(agent, env)
    agent.label = 'MC First (gamma=1)'
    env, agent = interactive(env, agent)
    env.view_mode = 1 # Automatically set value-function view-mode.
    # env = VideoMonitor(env, agent=agent, fps=200, render_kwargs={'method_label': 'MC first'})
    train(env, agent, num_episodes=1000)
    env.close()
