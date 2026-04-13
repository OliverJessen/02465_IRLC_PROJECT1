# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from irlc.utils.common import defaultdict2
from irlc import Agent

class TabularAgent(Agent):
    """
    This helper class will simplify the implementation of most basic reinforcement learning. Specifically it provides:

        - A :math:`Q(s,a)`-table data structure
        - An epsilon-greedy exploration method

    The code for the class is very simple, and I think it is a good idea to at least skim it.

    The Q-data structure can be used a follows:

    .. runblock:: pycon

        >>> from irlc.ex09.rl_agent import TabularAgent
        >>> from irlc.gridworld.gridworld_environments import BookGridEnvironment
        >>> env = BookGridEnvironment()
        >>> agent = TabularAgent(env)
        >>> state, info = env.reset()               # Get the info-dictionary corresponding to s
        >>> agent.Q[state, 1] = 2.5                 # Update a Q-value; action a=1 is now optimal.
        >>> agent.Q[state, 1]                       # Check it has indeed been updated.
        >>> agent.Q[state, 0]                       # Q-values are 0 by default.
        >>> agent.Q.get_optimal_action(state, info) # Note we pass along the info-dictionary corresopnding to this state

    .. note::
        The ``get_optimal_action``-function requires an ``info`` dictionary. This is required since the info dictionary
        contains information about which actions are available. To read more about the Q-values, see :class:`~irlc.ex09.rl_agent.TabularQ`.
    """
    def __init__(self, env, gamma=0.99, epsilon=0):
        r"""
        Initialize a tabular environment. For convenience, it stores the discount factor :math:`\gamma` and
        exploration parameter :math:`\varepsilon` for epsilon-greedy exploration. Access them as e.g. ``self.gamma``

        When you implement an agent and overwrite the ``__init__``-method, you should include a call such as
        ``super().__init__(gamma, epsilon)``.

        :param env:  The gym environment
        :param gamma: The discount factor :math:`\gamma`
        :param epsilon: Exploration parameter :math:`\varepsilon` for epsilon-greedy exploration
        """
        super().__init__(env)
        self.gamma, self.epsilon = gamma, epsilon
        self.Q = TabularQ(env)

    def pi_eps(self, s, info):
        """
        Performs :math:`\\varepsilon`-greedy exploration with :math:`\\varepsilon =` ``self.epsilon`` and returns the
        action. Recall this means that with probability :math:`\\varepsilon` it returns a random action, and otherwise
        it returns an action associated with a maximal Q-value (:math:`\\arg\\max_a Q(s,a)`). An example:

        .. runblock:: pycon

            >>> from irlc.ex09.rl_agent import TabularAgent
            >>> from irlc.gridworld.gridworld_environments import BookGridEnvironment
            >>> env = BookGridEnvironment()
            >>> agent = TabularAgent(env)
            >>> state, info = env.reset()
            >>> agent.pi_eps(state, info) # Note we pass along the info-dictionary corresopnding to this state

        .. note::
            The ``info`` dictionary is used to mask (exclude) actions that are not possible in the state.
            It is similar to the info dictionary in ``agent.pi(s,info)``.

        :param s: A state :math:`s_t`
        :param info: The corresponding ``info``-dictionary returned by the gym environment
        :return: An action computed using :math:`\\varepsilon`-greedy action selection based the Q-values stored in the ``self.Q`` class.
        """
        if info is not None and 'seed' in info: # In case info contains a seed, reset the random number generator.
            np.random.seed(info['seed'])
        return Agent.pi(self, s, k=0, info=info) if np.random.rand() < self.epsilon else self.Q.get_optimal_action(s, info)


class ValueAgent(TabularAgent): 
    """
    This is a simple wrapper class around the Agent class above. It fixes the policy and is therefore useful for doing
    value estimation.
    """
    def __init__(self, env, gamma=0.95, policy=None, v_init_fun=None): 
        self.env = env
        self.policy = policy  # policy to evaluate
        """ self.v holds the value estimates. 
        Initially v[s] = 0 unless v_init_fun is given in which case v[s] = v_init_fun(s). """
        self.v = defaultdict2(float if v_init_fun is None else v_init_fun) 
        super().__init__(env, gamma=gamma)
        self.Q = None  # Blank out the Q-values which will not be used.

    def pi(self, s, k, info=None):
        return TabularAgent.pi(self, s, k, info) if self.policy is None else self.policy(s) 

    def value(self, s):
        return self.v[s]

def _masked_actions(action_space, mask):
    """Helper function which applies a mask to the action space."""
    from irlc.utils.common import DiscreteTextActionSpace
    if isinstance(action_space, DiscreteTextActionSpace):
        return [a for a in range(action_space.n) if mask[a] == 1]
    else:
        return [a for a in range(action_space.n) if mask[a - action_space.start] == 1]


class TabularQ:
    """
    This is a helper class for storing Q-values. It is used by the :class:`~ircl.ex09.rl_agent.TabularAgent` to store
    Q-values where it can be be accessed as ``self.Q[s,a]``.
    """
    def __init__(self, env):
        """
        Initialize the table. It requires a gym environment to know how many actions there are for each state.
        :param env: A gym environment.
        """
        self._known_masks = {} # Cache the known action masks.

        def q_default(s):
            if s in self._known_masks:
                return {a: 0 for a in range(self.env.action_space.n) if self._known_masks[s][a- self.env.action_space.start] == 1}
            else:
                return {a: 0 for a in range(self.env.action_space.n)}

        self.q_ = defaultdict2(lambda s: q_default(s))
        self.env = env

    def get_Qs(self, state, info_s=None):
        """
        Get a list of all known Q-values for this particular state. That is, in a given state, it will return the two
        lists:

        .. math::
            \\begin{bmatrix} a_1 \\\\ a_2 \\\\ \\vdots \\\\ a_k \\end{bmatrix},  \\quad
            \\begin{bmatrix} Q(s,a_1) \\\\ Q(s,a_1) \\\\ \\vdots \\\\ Q(s,a_k) \\end{bmatrix} \\\\

        the ``info_s`` parameter will ensure actions are correctly masked. An example of how to use this function from
        a policy:

        .. runblock:: pycon

            >>> from irlc.ex09.rl_agent import TabularAgent
            >>> class MyAgent(TabularAgent):
            ...     def pi(self, s, k, info=None):
            ...         actions, q_values = self.Q.get_Qs(s, info)

        :param state: The state to query
        :param info_s: The info-dictionary returned by the environment for this state. Used for action-masking.
        :return:
            - actions - A tuple containing all actions available in this state ``(a_1, a_2, ..., a_k)``
            - Qs - A tuple containing all Q-values available in this state ``(Q[s,a1], Q[s, a2], ..., Q[s,ak])``
        """
        if info_s is not None and 'mask' in info_s:
            if state not in self._known_masks:
                self._known_masks[state] = info_s['mask']
                # Probably a good idea to check the Q-values are okay...
                avail_actions = _masked_actions(self.env.action_space, info_s['mask'])
                self.q_[state] = {a: self.q_[state][a] for a in avail_actions}

        (actions, Qa) = zip(*self.q_[state].items())
        return tuple(actions), tuple(Qa)

    def get_optimal_action(self, state, info_s):
        """
        For a given state ``state``, this function returns the optimal action for that state.

        .. math::
            a^* = \\arg\\max_a Q(s,a)

        An example:
        .. runblock:: pycon

            >>> from irlc.ex09.rl_agent import TabularAgent
            >>> class MyAgent(TabularAgent):
            ...     def pi(self, s, k, info=None):
            ...         a_star = self.Q.get_optimal_action(s, info)


        :param state: State to find the optimal action in :math:`s`
        :param info_s: The ``info``-dictionary corresponding to this state
        :return: The optimal action according to the Q-table :math:`a^*`
        """
        actions, Qa = self.get_Qs(state, info_s)
        a_ = np.argmax(np.asarray(Qa) + np.random.rand(len(Qa)) * 1e-8)
        return actions[a_]

    def _chk_mask(self, s, a):
        if s in self._known_masks:
            mask = self._known_masks[s]
            if mask[a - self.env.action_space.start] == 0:
                raise Exception(f" Invalid action. You tried to access Q[{s}, {a}], however the action {a} has been previously masked and therefore cannot exist in this state. The mask for {s} is mask={mask}.")

    def __getitem__(self, state_comma_action):
        s, a = state_comma_action
        self._chk_mask(s, a)
        return self.q_[s][a]

    def __setitem__(self, state_comma_action, q_value):
        s, a = state_comma_action
        self._chk_mask(s, a)
        self.q_[s][a] = q_value

    def to_dict(self):
        """
        This helper function converts the known Q-values to a dictionary. This function is only used for
        visualization purposes in some of the examples.

        :return: A dictionary ``q`` of all known Q-values of the form ``q[s][a]``
        """
        # Convert to a regular dictionary
        d = {s: {a: Q for a, Q in Qs.items() } for s,Qs in self.q_.items()}
        return d
