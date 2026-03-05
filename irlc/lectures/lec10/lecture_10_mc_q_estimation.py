# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.gridworld.gridworld_environments import BookGridEnvironment
from irlc import train, interactive

def keyboard_play(env, agent, method_label='MC',autoplay=False, num_episodes=1000):
    agent.label = method_label
    env, agent = interactive(env, agent, autoplay=autoplay)
    # agent = PlayWrapper(agent, env,autoplay=autoplay)
    # env = VideoMonitor(env, agent=agent, fps=100, agent_monitor_keys=('pi', 'Q'), render_kwargs={'method_label': method_label})
    train(env, agent, num_episodes=num_episodes)
    env.close()


if __name__ == "__main__":
    env = BookGridEnvironment(render_mode='human', zoom=2, living_reward=-0.05)
    from irlc.ex09.mc_agent import MCAgent
    agent = MCAgent(env, gamma=0.9, epsilon=1., first_visit=True, alpha=None)
    # agent.label =
    # env, agent = interactive(env, agent)
    keyboard_play(env, agent, method_label='MC Q-estimation (First visit)')
