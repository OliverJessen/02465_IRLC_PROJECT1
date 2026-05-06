# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
import random
from collections import deque
from irlc import cache_read, cache_write

class BasicBuffer:
    """
    The buffer class is used to keep track of past experience and sample it for learning.
    """
    def __init__(self, max_size=2000):
        """
        Creates a new (empty) buffer.

        :param max_size: Maximum number of elements in the buffer. This should be a large number like 100'000.
        """
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        """
        Add information from a single step, :math:`(s_t, a_t, r_{t+1}, s_{t+1}, \\text{done})` to the buffer.

        .. runblock:: pycon

            >>> import gymnasium as gym
            >>> from irlc.ex13.buffer import BasicBuffer
            >>> env = gym.make("CartPole-v1")
            >>> b = BasicBuffer()
            >>> s, info = env.reset()
            >>> a = env.action_space.sample()
            >>> sp, r, done, _, info = env.step(a)
            >>> b.push(s, a, r, sp, done)
            >>> len(b) # Get number of elements in buffer

        :param state: A state :math:`s_t`
        :param action: Action taken :math:`a_t`
        :param reward: Reward obtained :math:`r_{t+1}`
        :param next_state: Next state transitioned to :math:`s_{t+1}`
        :param done: ``True`` if the environment terminated else ``False``
        :return: ``None``
        """
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample ``batch_size`` elements from the buffer for use in training a deep Q-learning method.
        The elements returned all be numpy ``ndarray`` where the first dimension is the batch dimension, i.e. of size
        ``batch_size``.

        .. runblock:: pycon

            >>> import gymnasium as gym
            >>> from irlc.ex13.buffer import BasicBuffer
            >>> env = gym.make("CartPole-v1")
            >>> b = BasicBuffer()
            >>> s, info = env.reset()
            >>> a = env.action_space.sample()
            >>> sp, r, done, _, _ = env.step(a)
            >>> b.push(s, a, r, sp, done)
            >>> S, A, R, SP, DONE = b.sample(batch_size=32)
            >>> S.shape # Dimension batch_size x n
            >>> R.shape # Dimension batch_size x 1

        :param batch_size: Number of elements to sample
        :return:
            - S - Matrix of size ``batch_size x n`` of sampled states
            - A - Matrix of size ``batch_size x n`` of sampled actions
            - R - Matrix of size ``batch_size x n`` of sampled rewards
            - SP - Matrix of size ``batch_size x n`` of sampled states transitioned to
            - DONE - Matrix of size ``batch_size x 1`` of bools indicating if the environment terminated

        """
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        assert len(self.buffer) > 0, "The replay buffer must be non-empty in order to sample a batch: Use push()"
        batch = random.choices(self.buffer, k=batch_size)
        for state, action, reward, next_state, done in batch:
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return map(lambda x: np.asarray(x), (state_batch, action_batch, reward_batch, next_state_batch, done_batch))

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        """
        Use this to save the content of the buffer to a file

        :param path: Path where to save (use same argument with ``load``)
        :return: ``None``
        """
        cache_write(self.buffer, path)

    def load(self, path):
        """
        Use this to load buffer content from a file

        :param path: Path to load from (use same argument with ``save``)
        :return: ``None``
        """
        self.buffer = cache_read(path)
