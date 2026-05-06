# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
class DQNNetwork:
    """
    A class representing a deep Q network.
    Note that this function is batched. I.e. ``s`` is assumed to be a numpy array of dimension ``batch_size x n``

    The following example shows how you can evaluate the Q-values in a given state. An example:

    .. runblock:: pycon

        >>> from irlc.ex13.torch_networks import TorchNetwork
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> env = gym.make("CartPole-v1")
        >>> Q = TorchNetwork(env, trainable=True, learning_rate=0.001) # DQN network requires an env to set network dimensions
        >>> batch_size = 32 # As an example
        >>> states = np.random.rand(batch_size, env.observation_space.shape[0]) # Creates some dummy input
        >>> states.shape    # batch_size x n
        >>> qvals = Q(states) # Evaluate Q(s,a)
        >>> qvals.shape # This is a tensor of dimension batch_size x actions
        >>> print(qvals[0,1]) # Get Q(s_0, 1)
        >>> Y = np.random.rand(batch_size, env.action_space.n) # Generate target Q-values (training data)
        >>> Q.fit(states, Y)                      # Train the Q-network for 1 gradient descent step
    """
    def update_Phi(self, source, tau=0.01):
        r"""
        Update (adapts) the weights in this network towards those in source by a small amount.

        For each weight :math:`w_i` in (this) network, and each corresponding weight :math:`w'_i` in the ``source`` network,
        the following Polyak update is performed:

        .. math::
            w_i \leftarrow w_i + \tau (w'_i - w_i)

        :param source: Target network to update towards
        :param tau: Update rate (rate of change :math:`\\tau`
        :return: ``None``
        """

        raise NotImplementedError

    def __call__(self, s):
        """
        Evaluate the Q-values in the given (batched) state.

        :param s: A matrix of size ``batch_size x n`` where :math:`n` is the state dimension.
        :return: The Q-values as a ``batch_size x d`` dimensional matrix where :math:`d` is the number of actions.
        """
        raise NotImplementedError

    def fit(self, s, target): 
        r"""
        Fit the network weights by minimizing

        .. math::
            \frac{1}{B}\sum_{i=1}^B \sum_{a=1}^K \| q_\phi(s_i)_a - y_{i,a} \|^2

        where ``target`` corresponds to :math:`y` and is a ``[batch_size x actions]`` matrix of target Q-values.
        :param s: 
        :param target: 
        :return: 
        """
        raise NotImplementedError
