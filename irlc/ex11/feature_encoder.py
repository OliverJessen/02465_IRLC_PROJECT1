# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (Freely available online).
"""
from math import floor
from gymnasium.spaces.box import Box
import numpy as np
from irlc.ex08.rl_agent import _masked_actions
from irlc.utils.common import defaultdict2

class FeatureEncoder:
    r"""
    The idea behind linear function approximation of :math:`Q`-values is that

    - We initialize (and eventually learn) a :math:`d`-dimensional weight vector :math:`w \in \mathbb{R}^d`
    - We assume there exists a function to compute a :math:`d`-dimensional feature vector :math:`x(s,a) \in \mathbb{R}^d`
    - The :math:`Q`-values are then represented as

      .. math::
         Q(s,a) = x(s,a)^\top w

    Learning is therefore entirely about updating :math:`w`.

    The following example shows how you initialize the linear :math:`Q`-values and compute them in a given state:

    .. runblock:: pycon

        >>> import gymnasium as gym
        >>> from irlc.ex11.feature_encoder import LinearQEncoder
        >>> env = gym.make('MountainCar-v0')
        >>> Q = LinearQEncoder(env, tilings=8)
        >>> s, _ = env.reset()
        >>> a = env.action_space.sample()
        >>> Q(s,a) # Compute a Q-value.
        >>> Q.d             # Get the number of dimensions
        >>> Q.x(s,a)[:4]    # Get the first four coordinates of the x-vector
        >>> Q.w[:4]         # Get the first four coordinates of the w-vector

    """
    def __init__(self, env):
        """
        Initialize the feature encoder. It requires an environment to know the number of actions and dimension of the state space.

        :param env: An openai Gym ``Env``.
        """
        self.env = env
        self.w = np.zeros((self.d, ))
        self._known_masks = {}

        def q_default(s):
            from irlc.utils.common import DiscreteTextActionSpace
            if s in self._known_masks:
                return {a: 0 for a in range(self.env.action_space.n) if
                        self._known_masks[s][(a - self.env.action_space.start) if not isinstance(self.env.action_space, DiscreteTextActionSpace) else a] == 1}
            else:
                return {a: 0 for a in range(self.env.action_space.n)}

        self.q_ = defaultdict2(lambda s: q_default(s))

    @property
    def d(self):
        """ Get the number of dimensions of :math:`w`

        .. runblock:: pycon

            >>> import gymnasium as gym
            >>> from irlc.ex11.feature_encoder import LinearQEncoder
            >>> env = gym.make('MountainCar-v0')
            >>> Q = LinearQEncoder(env, tilings=8) # Same encoding as Sutton & Barto
            >>> Q.d
        """
        raise NotImplementedError()

    def x(self, s, a):
        """
        Computes the :math:`d`-dimensional feature vector :math:`x(s,a)`

        .. runblock:: pycon

           >>> import gymnasium as gym
           >>> from irlc.ex11.feature_encoder import LinearQEncoder
           >>> env = gym.make('MountainCar-v0')
           >>> Q = LinearQEncoder(env, tilings=8) # Same encoding as Sutton & Barto
           >>> s, info = env.reset()
           >>> x = Q.x(s, env.action_space.sample())

        :param s: A state :math:`s`
        :param a: An action :math:`a`
        :return: Feature vector :math:`x(s,a)`
        """
        raise NotImplementedError()

    def get_Qs(self, state, info_s=None):
        """
        This is a helper function, it is only for internal use.

        :param state:
        :param info_s:
        :return:
        """
        if info_s is not None and 'mask' in info_s and not isinstance(state, np.ndarray):
            if state not in self._known_masks:
                self._known_masks[state] = info_s['mask']
                # Probably a good idea to check the Q-values are okay...
                avail_actions = _masked_actions(self.env.action_space, info_s['mask'])
                self.q_[state] = {a: self.q_[state][a] for a in avail_actions}

        from irlc.pacman.pacman_environment import PacmanEnvironment
        from irlc.pacman.pacman_utils import Actions
        if isinstance(state, np.ndarray):
            actions = tuple(range(self.env.action_space.n))
        elif isinstance(self.env, PacmanEnvironment):
            # actions = Actions
            # actions = tuple(Actions._directions.keys())
            actions =  _masked_actions(self.env.action_space, info_s['mask'])
            actions = tuple([self.env.action_space.actions[n] for n in actions])
        else:
            actions = tuple(self.q_[state].keys())

        # if isinstance(self.env, PacmanEnvironment):
        #     # TODO: Make smarter masking.
        #     actions = [a for a in actions if a in self.env.A(state)]
        # actions =
        Qs = tuple([self(state,a) for a in actions])
        # TODO: Implement masking and masking-cache.
        return actions, Qs
        #
        # actions = list( self.env.P[state].keys() if hasattr(self.env, 'P') else range(self.env.action_space.n) )
        # Qs = [self(state, a) for a in actions]
        # return tuple(actions), tuple(Qs)

    def get_optimal_action(self, state, info=None):
        r"""
        For a given state ``state``, this function returns the optimal action for that state.

        .. math::
            a^* = \arg\max_a Q(s,a)

        An example:

        .. runblock:: pycon

           >>> from irlc.ex08.rl_agent import TabularAgent
           >>> class MyAgent(TabularAgent):
           ...     def pi(self, s, k, info=None):
           ...         a_star = self.Q.get_optimal_action(s, info)

        :param state: State to find the optimal action in :math:`s`
        :param info: The ``info``-dictionary corresponding to this state
        :return: The optimal action according to the Q-values :math:`a^*`
        """
        actions, Qa = self.get_Qs(state, info)
        if len(actions) == 0:
            print("Bad actions list")
        a_ = np.argmax(np.asarray(Qa) + np.random.rand(len(Qa)) * 1e-8)
        return actions[a_]

    def __call__(self, s, a):
        """
        Evaluate the Q-values for the given state and action. An example:

        .. runblock:: pycon

           >>> import gymnasium as gym
           >>> from irlc.ex11.feature_encoder import LinearQEncoder
           >>> env = gym.make('MountainCar-v0')
           >>> Q = LinearQEncoder(env, tilings=8) # Same encoding as Sutton & Barto
           >>> s, info = env.reset()
           >>> Q(s, env.action_space.sample()). # Compute Q(s,a)

        :param s: A state :math:`s`
        :param a: An action :math:`a`
        :return: Feature vector :math:`x(s,a)`
        """
        return self.x(s, a) @ self.w

    def __getitem__(self, item):
        raise Exception("Hi! You tried to access linear Q-values as Q[s,a]. You need to use Q(s,a). This choice signifies they are not represented as a table, but as a linear combination x(s,a)^T w")
        # s,a = item
        # return self.__call__(s, a)

    def __setitem__(self, key, value):
        raise Exception("Oy! You tried to set a linearly encoded Q-value as in Q[s, a] = new_q_value.\n This is not possible since they are represented as x(s,a)^T w. Rewrite the expression to update Q.w.")

class DirectEncoder(FeatureEncoder):
    def __init__(self, env):
        self.d_ = np.prod( env.observation_space.shape ) * env.action_space.n
        # self.d_ = len(self.x(env.reset(), env.action_space.n))
        super().__init__(env)

    def x(self, s, a):
        xx = np.zeros( (self.d,))
        n = s.size
        xx[n * a:n*(a+1) ] = s
        return xx

        ospace = self.env.observation_space.shape
        simple = False
        if not isinstance(ospace, tuple):
            ospace = (ospace,)
            simple = True

        sz = []
        for j, disc in enumerate(ospace):
            sz.append(disc.n)

        total_size = sum(sz)
        csum = np.cumsum(sz, ) - sz[0]
        self.max_size = total_size * self.env.action_space.n


        def fixed_sparse_representation(s, action):
            if simple:
                s = (s,)
            s_encoded = [cs + ds + total_size * action for ds, cs in zip(s, csum)]
            return s_encoded

        self.get_active_tiles = fixed_sparse_representation

    # super().__init__(env)

    @property
    def d(self):
        return self.d_
        return 10000*8
        x = np.zeros(self.d)
        at = self.get_active_tiles(s, a)
        x[at] = 1.0
        return x


class GridworldXYEncoder(FeatureEncoder):
    def __init__(self, env):
        self.env = env
        self.na = self.env.action_space.n
        self.ns = 2
        super().__init__(env)

    @property
    def d(self):
        return self.na*self.ns

    def x(self, s, a):
        x,y = s
        xx = [np.zeros(self.ns) for _ in range(self.na)]
        xx[a][0] = x
        xx[a][1] = y
        # return xx[a]
        xx = np.concatenate(xx)
        return xx

class SimplePacmanExtractor(FeatureEncoder):
    def __init__(self, env):
        self.env = env
        from irlc.pacman.feature_extractor import SimpleExtractor
        # from reinforcement.featureExtractors import SimpleExtractor
        self._extractor = SimpleExtractor()
        self.fields = ["bias", "#-of-ghosts-1-step-away", "#-of-ghosts-1-step-away", "eats-food", "closest-food"]
        super().__init__(env)

    def x(self, s, a):
        xx = np.zeros_like(self.w)
        # ap = self.env._actions_gym2pac[a]
        ap = a
        for k, v in self._extractor.getFeatures(s, ap).items():
            xx[self.fields.index(k)] = v
        return xx

    @property
    def d(self):
        return len(self.fields)

class LinearQEncoder(FeatureEncoder):
    def __init__(self, env, tilings=8, max_size=2048):
        r"""
        Implements the tile-encoder described by (SB18)

        :param env: The openai Gym environment we wish to solve.
        :param tilings: Number of tilings (translations). Typically 8.
        :param max_size: Maximum number of dimensions.
        """
        if isinstance(env.observation_space, Box):
            os = env.observation_space
            low = os.low
            high = os.high
            scale = tilings / (high - low)
            hash_table = IHT(max_size)
            self.max_size = max_size
            def tile_representation(s, action):
                s_ = list( (s*scale).flat )
                active_tiles = tiles(hash_table, tilings, s_, [action]) # (s * scale).tolist()
                # if 0 not in active_tiles:
                #     active_tiles.append(0)
                return active_tiles
            self.get_active_tiles = tile_representation
        else:
            # raise Exception("Implement in new class")
            #
            # Use Fixed Sparse Representation. See:
            # https://castlelab.princeton.edu/html/ORF544/Readings/Geramifard%20-%20Tutorial%20on%20linear%20function%20approximations%20for%20dynamic%20programming%20and%20RL.pdf

            ospace = env.observation_space
            simple = False
            if not isinstance(ospace, tuple):
                ospace = (ospace,)
                simple = True

            sz = []
            for j,disc in enumerate(ospace):
                sz.append( disc.n )

            total_size = sum(sz)
            csum = np.cumsum(sz,) - sz[0]
            self.max_size = total_size * env.action_space.n

            def fixed_sparse_representation(s, action):
                if simple:
                   s = (s,)
                s_encoded = [cs + ds + total_size * action for ds,cs in zip(s, csum)]
                return s_encoded
            self.get_active_tiles = fixed_sparse_representation
        super().__init__(env)

    def x(self, s, a):
        x = np.zeros(self.d)
        at = self.get_active_tiles(s, a)
        x[at] = 1.0
        return x

    @property
    def d(self):
        return self.max_size


"""
Following code contains the tile-coding utilities copied from:
http://incompleteideas.net/tiles/tiles3.py-remove
"""
class IHT:
    """Structure to handle collisions"""

    def __init__(self, size_val):
        self.size = size_val
        self.overfull_count = 0
        self.dictionary = {}


    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull_count == 0:
                print('IHT full, starting to allow collisions')
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count


def hash_coords(coordinates, m, read_only=False):
    if isinstance(m, IHT): return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int): return hash(tuple(coordinates)) % m
    if m is None: return coordinates



def tiles(iht_or_size, num_tilings, floats, ints=None, read_only=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    if ints is None:
        ints = []
    qfloats = [floor(f * num_tilings) for f in floats]
    tiles = []
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        tiles.append(hash_coords(coords, iht_or_size, read_only))
    return tiles
