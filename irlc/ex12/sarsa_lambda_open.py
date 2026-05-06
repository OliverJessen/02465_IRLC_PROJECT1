# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex12.sarsa_lambda_agent import SarsaLambdaAgent
from irlc.gridworld.gridworld_environments import OpenGridEnvironment
from irlc import train, interactive

def keyboard_play(Agent, method_label='MC', num_episodes=1000, alpha=0.5, autoplay=False, **args):
    print("Evaluating", Agent, "on the open gridworld environment.")
    print("Press p to follow the agents policy or use the keyboard to input actions")
    print("(Please be aware that Sarsa, N-step Sarsa, and Sarsa(Lambda) do not always make the right updates when you input actions with the keyboard)")

    env = OpenGridEnvironment(render_mode='human', frames_per_second=10)
    try:
        agent = Agent(env, gamma=0.99, epsilon=0.1, alpha=alpha, **args)
    except Exception as e: # If it is a value agent without the epsilon.
        agent = Agent(env, gamma=0.99, alpha=alpha, **args)
    env, agent = interactive(env, agent, autoplay=autoplay)
    train(env, agent, num_episodes=num_episodes)
    env.close()

if __name__ == "__main__":
    """ 
    Example: Play a three episodes and save a snapshot of the Q-values as a .pdf
    """
    env = OpenGridEnvironment(render_mode='human')
    agent = SarsaLambdaAgent(env, gamma=0.99, epsilon=0.1, alpha=.5)
    env, agent = interactive(env, agent, autoplay=True)
    train(env, agent, num_episodes=3)
    from irlc import savepdf
    savepdf("sarsa_lambda_opengrid", env=env)
    env.close()

    """ Example: Keyboard play 
    You can input actions manually with the keyboard, but the Q-values are not necessarily updates correctly in this mode. Can you tell why? 
    You can let the agent play by pressing `p`, in which case the Q-values will be updated correctly. """
    keyboard_play(SarsaLambdaAgent, method_label="Sarsa(Lambda)", lamb=0.8)
