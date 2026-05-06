# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from irlc import train, interactive, savepdf
from irlc.gridworld.gridworld_environments import GridworldEnvironment, grid_bridge_grid
from irlc.project4.rebels import very_basic_grid
from irlc.ex10.q_agent import QAgent
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('qtagg')


if __name__ == "__main__":
    np.random.seed(42) # Fix the seed for reproduciability
    env = GridworldEnvironment(very_basic_grid, render_mode='human') # Create an environment
    env.reset()                   # Reset (to set up the visualization)
    savepdf("rebels_basic", env=env)   # Save a snapshot of the starting state
    env.close()

    # Create an interactive version.
    env = GridworldEnvironment(very_basic_grid, render_mode='human')  # Create an environment
    agent = QAgent(env) # This agent will display the Q-values.
    # agent = Agent(env) # A random agent.
    # env, agent = interactive(env, agent) # Uncomment this line to play in 'env' environment. Use space to let the agent move.
    stats, trajectories = train(env, agent, num_episodes=16, return_trajectory=True)
    env.close()
    print("Trajectory 0: States traversed", trajectories[0].state, "actions taken", trajectories[0].action) 
    print("Trajectory 1: States traversed", trajectories[1].state, "actions taken", trajectories[1].action)
    all_actions = [t.action[:-1] for t in trajectories] # Concatenate all action sequence excluding the last dummy-action.
    print("All actions taken in 16 episodes, excluding the terminal (dummy) action", all_actions) 
    # Note the last list is of length 20 -- this is because the environment will always terminate after two actions,
    # and since we discard the last (dummy) action we get 20 actions.
    # In general, the list of actions will be longer, as only the last action should be discarded (as in the code above).

    # A more minimalistic example to plot the bridge-grid environment
    bridge_env = GridworldEnvironment(grid_bridge_grid, render_mode='human')
    bridge_env.reset()
    savepdf("rebels_bridge", env=bridge_env)
    bridge_env.close()

    # The following code will simulate a Q-learning agent for 3000 (!) episodes and plot the Q-functions.
    np.random.seed(42)  # Fix the seed for reproduciability
    env = GridworldEnvironment(grid_bridge_grid)
    agent = QAgent(env, alpha=0.1, epsilon=0.2, gamma=1)
    """ Uncomment the next line to play in the environment. 
    Use the space-bar to let the agent take an action, p to unpause, and otherwise use the keyboard arrows """
    train(env, agent, num_episodes=3000) # Train for 3000 episodes. Surely the rebels must be found by now!
    bridge_env, agent = interactive(env, agent)
    bridge_env.reset()
    bridge_env.savepdf("rebels_bridge_Q")
    bridge_env.close()
