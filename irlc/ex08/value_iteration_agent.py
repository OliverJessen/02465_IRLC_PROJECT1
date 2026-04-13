# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex08.value_iteration import value_iteration
from irlc import TabularAgent
import numpy as np


class ValueIterationAgent(TabularAgent):
    def __init__(self, env, mdp=None, gamma=1, epsilon=0, **kwargs):
        super().__init__(env)
        self.epsilon = epsilon
        # TODO: 1 lines missing.
        raise NotImplementedError("Call the value_iteration function and store the policy for later.")

    def pi(self, s, k, info=None):
        """ With probability (1-epsilon), the take optimal action as computed using value iteration
         With probability epsilon, take a random action. You can do this using return self.random_pi(s)
        """
        if np.random.rand() < self.epsilon:
            return super().pi(s, k, info) # Recall that by default the policy takes random actions.
        else:
            """ Return the optimal action here. This should be computed using value-iteration. 
             To speed things up, I recommend calling value-iteration from the __init__-method and store the policy. """
            # TODO: 1 lines missing.
            raise NotImplementedError("Compute and return optimal action according to value-iteration.")
            return action

    def __str__(self):
        return f"ValueIteration(epsilon={self.epsilon})"


if __name__ == "__main__":
    from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment
    env = SuttonCornerGridEnvironment(living_reward=-1, render_mode='human')
    from irlc import train, interactive
    # Note you can access the MDP for a gridworld using env.mdp. The mdp will be an instance of the MDP class we have used for planning so far.
    agent = ValueIterationAgent(env, mdp=env.mdp) # Make a ValueIteartion-based agent
    # Visualize & interactivity. Press P or space to follow the policy.
    agent.Q = None # This ensures that the value function is visualized.
    env, agent = interactive(env, agent)
    train(env, agent, num_episodes=20)                             # Train for 100 episodes
    env.savepdf("smallgrid.pdf") # Take a snapshot of the final configuration
    env.close() # Whenever you use a VideoMonitor, call this to avoid a dumb openglwhatever error message on exit
