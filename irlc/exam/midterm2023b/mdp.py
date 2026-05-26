# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf).
"""
from tqdm import tqdm
import sys

class MDP: 
    """
    This class represents a Markov Decision Process. It defines three main components:

        - The actions available in a given state :math:`A(s)`
        - The transition probabilities :math:`p(s', r | s, a)`
        - A terminal check to determine if a state :math:`s` is terminal
        - A way to specify the initial state:

            - As a single state the MDP always begins in (most common)
            - As a general distribution :math:`p(s_0)`.

    In addition to this it allows you to access either
        - The set of all states (including terminal states) as ``mdp.states``
        - The set of all non-terminal states as ``mdp.non_terminal_states``

    .. note::
        The ``states`` and ``non_termianl_states`` are computed lazily. This means that if you don't access them, they won't use memory.
        This allows you to specify MDPs with an infinite number of states without running out of memory.
    """
    def __init__(self, initial_state=None, verbose=False):
        """
        Initialize the MDP. In the case where ``initial_state`` is set to a value :math:`s_0`, the initial state distribution will be

        .. math::
            p(s_0) = 1

        :param initial_state: An optional initial state.
        :param verbose: If ``True``, the class will print out debug information (useful for very large MDPs)
        """
        self.verbose=verbose
        self.initial_state = initial_state  # Starting state s_0 of the MDP. 
        # The following variables that begin with _ are used to cache computations. The reason why we don't compute them
        # up-front is because their computation may be time-consuming and they might not be needed.
        self._states = None
        self._nonterminal_states = None
        self._terminal_states = None

    def is_terminal(self, state): 
        """
        Determines if a state is terminal (i.e., the environment/model is complete). In (SB18), the terminal
        state is written as :math:`s_T`.

        .. runblock:: pycon

            >>> from irlc.gridworld.gridworld_environments import FrozenLake
            >>> mdp = FrozenLake().mdp
            >>> mdp.is_terminal(mdp.initial_state) # False, obviously.


        :param state: The state :math:`s` to check
        :return: ``True`` if the state is terminal and otherwise ``False``.
        """
        return False # Return true if the given state is terminal.

    def Psr(self, state, action):
        """
        Represents the transition probabilities:

        .. math::
            P(s', r | s, a)

        When called with state ``state`` and action ``action``, the function returns a dictionary of the form
        ``{(s1, r1): p1, (s2, r2): p2, ...}``, so that ``p2`` is the probability of transitioning to ``s2`` (and obtaining
        reward ``r2``) given we are in state ``state`` and take action ``action``:

        .. math::
            P(s_2, r_2 | s,a) = p_2

        An example:

        .. runblock:: pycon

            >>> from irlc.gridworld.gridworld_environments import FrozenLake
            >>> mdp = FrozenLake().mdp
            >>> transitions = mdp.Psr(mdp.initial_state, 0) # P( ... | s0, a=0)
            >>> for (sp, r), p in transitions.items():
            ...     print(f"P(s'={sp}, r={r} | s={mdp.initial_state}, a=0) = {p}")

        :param state: The state to compute the transition probabilities in
        :param action:  The action to compute the transition probabilities in
        :return: A dictionary where the keys are state, reward pairs we will transition to, :math:`p(s', r | ...)`, and the values are their probability.
        """
        raise NotImplementedError("Return state distribution as a dictionary (see class documentation)")

    def A(self, state):
        """
        Returns a list of actions available in the given state:

        .. math::
            A(s)

        An example to get the actions in the initial state:

        .. runblock:: pycon

            >>> from irlc.gridworld.gridworld_environments import FrozenLake
            >>> mdp = FrozenLake().mdp
            >>> mdp.A(mdp.initial_state)

        :param state: State to compute the actions in :math:`s`
        :return: The list of available actions :math:`\mathcal A(s) = \{0, 1, ..., n-1\}`
        """
        raise NotImplementedError("Return set/list of actions in given state A(s) = {a1, a2, ...}") 

    def initial_state_distribution(self):
        """
        (**Optional**) specify the initial state distribution. Should return a dictionary of the form:
        ``{s0: p0, s1: p1, ..., sn: pn}``, in which case :math:`p(S_0 = s_k) = p_k`.

        You will typically not overwrite this function but just set the initial state. In that case the initial state distribution
        is deterministic:


        .. runblock:: pycon

            >>> from irlc.gridworld.gridworld_environments import FrozenLake
            >>> mdp = FrozenLake().mdp
            >>> mdp.initial_state_distribution()



        :return: An initial state distribution as a dictionary, where the keys are states, and the valuse are their probability.
        """
        if self.initial_state is not None:
            return {self.initial_state: 1}
        else:
            raise Exception("Either specify the initial state, or implement this method.")

    @property
    def nonterminal_states(self):
        """
        The list of non-terminal states, i.e. :math:`\mathcal{S}` in (SB18)


        .. runblock:: pycon

            >>> from irlc.gridworld.gridworld_environments import FrozenLake
            >>> mdp = FrozenLake().mdp
            >>> mdp.nonterminal_states

        :return: The list of non-terminal states :math:`\mathcal{S}`
        """
        if self._nonterminal_states is None:
            self._nonterminal_states = [s for s in self.states if not self.is_terminal(s)]
        return self._nonterminal_states

    @property
    def states(self):
        """
        The list of all states including terminal ones, i.e. :math:`\mathcal{S}^+` in (SB18).
        The terminal states are those where ``is_terminal(state)`` is true.

        .. runblock:: pycon

            >>> from irlc.gridworld.gridworld_environments import FrozenLake
            >>> mdp = FrozenLake().mdp
            >>> mdp.states

        :return: The list all states :math:`\mathcal{S}^+`
        """
        if self._states is None:
            next_chunk = set(self.initial_state_distribution().keys())
            all_states = list(next_chunk)
            while True:
                new_states = set()
                for s in tqdm(next_chunk, file=sys.stdout) if self.verbose else next_chunk:
                    if self.is_terminal(s):
                        continue
                    for a in self.A(s):
                        new_states = new_states  | {sp for sp, r in self.Psr(s, a)}

                new_states  = [s for s in new_states if s not in all_states]
                if len(new_states) == 0:
                    break
                all_states += new_states
                next_chunk = new_states
            self._states = list(set(all_states))

        return self._states
