# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.gridworld.gridworld_environments import SuttonMazeEnvironment
from irlc import interactive, train
from irlc.gridworld.demo_agents.hidden_agents import ValueIterationAgent3

if __name__ == "__main__":

    # agent = ValueIterationAgent3(env, epsilon=0, gamma=1, only_update_current=False)
    env = SuttonMazeEnvironment(render_mode="human")
    agent = ValueIterationAgent3(env, epsilon=0, gamma=1, only_update_current=False)

    env, agent = interactive(env, agent)
    env.reset()
    train(env, agent, num_episodes=100)
    env.close()

    # sutton_maze_play(ValueIterationAgent3, method_label="DynaQ (n=5)", n=5)
