# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import UTestCase, Report
import numpy as np
import irlc
from irlc import train
from irlc.ex08.small_gridworld import SmallGridworldMDP
from irlc.ex08.policy_iteration import policy_iteration
from irlc.ex08.value_iteration import value_iteration
from irlc.gridworld.gridworld_environments import FrozenLake
from irlc.ex08.policy_evaluation import policy_evaluation

class Problem1_to_3_Warmup(UTestCase):
    def test_part1_average_reward(self):
        from irlc.ex08.mdp_warmup import expected_reward
        mdp = FrozenLake(living_reward=0.2).mdp  # Get the MDP of this environment.
        s0 = mdp.initial_state
        ## Part 1: Expected reward
        self.assertAlmostEqualC(expected_reward(mdp, s=s0, a=0), places=5)
        self.assertAlmostEqualC(expected_reward(mdp, s=s0, a=2), places=5)
        self.assertAlmostEqualC(expected_reward(mdp, s=(1,2), a=0), places=5)
        mdp = FrozenLake(living_reward=0.2).mdp  # Get the MDP of this environment.
        self.assertAlmostEqualC(expected_reward(mdp, s=s0, a=2), places=5)

    def test_part2_v2q(self):
        ## Part 2
        # First let's create a non-trivial value function
        V = {}
        mdp = FrozenLake(living_reward=0.3).mdp

        for k, s in enumerate(sorted(mdp.nonterminal_states)):
            V[s] = 2 * (s[0] - s[1]) - 3.5

        from irlc.ex08.mdp_warmup import value_function2q_function

        states = [(0, 1), (2, 3), (0, 3), (1,3), (1, 2)]

        s0 = mdp.initial_state

        q_ = value_function2q_function(mdp, s=s0, gamma=0.9, v=V)
        self.assertIsInstance(q_, dict)
        self.assertEqual(list(sorted(q_.keys())), [0, 1, 2, 3] )

        self.assertEqual(len(q_), 4)
        self.assertEqual(len(value_function2q_function(mdp, s=(1,2), gamma=0.9, v=V)), 1)
        self.assertAlmostEqualC(q_[0],places=4)
        self.assertAlmostEqualC(q_[2], places=4)


        for s in sorted(states):
            q_ = value_function2q_function(mdp, s=s, gamma=0.9, v=V)
            for a in [0, 1, 2, 3]:
                if a in mdp.A(s):
                    self.assertAlmostEqualC(q_[a], places=4)

    def test_part2_q2v(self):
        ## Part 3
        mdp = FrozenLake(living_reward=0.2).mdp
        from irlc.ex08.mdp_warmup import value_function2q_function, q_function2value_function
        # Create a non-trivial Q-function for this problem.
        Q = {}
        s0 = mdp.initial_state

        for k, s in enumerate(mdp.nonterminal_states):
            for a in mdp.A(s):
                Q[s, a] =  (s[0] - s[1]) - 5 * a  # The particular values are not important in this example
        # Create a policy. In this case pi(a=3) = 0.4.
        pi = {0: 0.2,
              1: 0.4,
              2: 0.2,
              3: 0.2}
        self.assertAlmostEqualC(q_function2value_function(pi, Q, s=s0), places=4)

def train_recording(env, agent, trajectories):
    for t in trajectories:
        env.reset()
        for k in range(len(t.action)):
            s = t.state[k]
            r = t.reward[k]
            a = t.action[k]
            sp = t.state[k+1]
            info = t.info[k]
            info_sp = t.info[k+1]

            agent.pi(s,k)
            agent.train(s, a, r, sp, done=k == len(t.action)-1, info_s = info, info_sp=info_sp)


class ValueFunctionTest(UTestCase):
    def check_value_function(self, mdp, V):
        self.assertL2(np.asarray([V[s] for s in mdp.states]), tol=1e-3)

class Problem5PolicyIteration(ValueFunctionTest):
    """ Iterative Policy iteration """
    def test_policy_iteration(self):
        env = SmallGridworldMDP()
        pi, v = policy_iteration(env, gamma=0.91)
        self.check_value_function(env, v)



class Problem6ValueIteration(ValueFunctionTest):
    """ Iterative value iteration """
    def test_value_iteration(self):
        env = SmallGridworldMDP()
        # from i
        pi, v = value_iteration(env, gamma=0.91)
        self.check_value_function(env, v)



class Problem4PolicyEvaluation(ValueFunctionTest):
    """ Iterative value iteration """
    def test_policy_evaluation(self):
        mdp = SmallGridworldMDP()
        pi = {s: {a: 1/len(mdp.A(s)) for a in mdp.A(s) } for s in mdp.nonterminal_states }
        v = policy_evaluation(pi, mdp, gamma=0.91)
        self.check_value_function(mdp, v)

    def test_policy_evaluation_b(self):
        mdp = SmallGridworldMDP()
        pi = {s: {a: 1 if a == 0 else 0 for a in mdp.A(s) } for s in mdp.nonterminal_states }
        v = policy_evaluation(pi, mdp, gamma=0.91)
        self.check_value_function(mdp, v)




class Problem9Gambler(ValueFunctionTest):
    """ Gambler's problem """
    def test_gambler_value_function(self):
        # from irlc.ex09.small_gridworld import SmallGridworldMDP, plot_value_function
        # from irlc.ex09.policy_iteration import policy_iteration
        # from irlc.ex09.value_iteration import value_iteration
        from irlc.ex08.gambler import GamblerMDP
        env = GamblerMDP()
        pi, v = value_iteration(env, gamma=0.91)
        self.check_value_function(env, v)

# class JackQuestion(ValueFunctionTest):
#     """ Gambler's problem """
#     def test_jacks_rental_value_function(self):
#         # from irlc.ex09.small_gridworld import SmallGridworldMDP, plot_value_function
#         # from irlc.ex09.policy_iteration import policy_iteration
#         # from irlc.ex09.value_iteration import value_iteration
#         # from irlc.ex09.gambler import GamblerEnv
#         from irlc.ex09.jacks_car_rental import JackRentalMDP
#         max_cars = 5
#         env = JackRentalMDP(max_cars=max_cars, verbose=True)
#         pi, V = value_iteration(env, gamma=.9, theta=1e-3, max_iters=1000, verbose=True)
#         self.check_value_function(env, V)

# class JackQuestion(QuestionGroup):
#     title = "Jacks car rental problem"
#
#     class JackItem(GridworldDPItem):
#         title = "Value function test"
#         max_cars = 5
#         tol = 0.01
#
#         def get_value_function(self):
#             from irlc.ex09.value_iteration import value_iteration
#             from irlc.ex09.jacks_car_rental import JackRentalMDP
#             env = JackRentalMDP(max_cars=self.max_cars, verbose=True)
#             pi, V = value_iteration(env, gamma=.9, theta=1e-3, max_iters=1000, verbose=True)
#             return V, env


        # return v, env
    # pass
# class DynamicalProgrammingGroup(QuestionGroup):
#     title = "Dynamical Programming test"
#
#     class PolicyEvaluationItem(GridworldDPItem):
#         title = "Iterative Policy evaluation"
#
#
#
#     class PolicyIterationItem(GridworldDPItem):
#         title = "policy iteration"
#         def get_value_function(self):
#             from irlc.ex09.small_gridworld import SmallGridworldMDP
#             from irlc.ex09.policy_iteration import policy_iteration
#             env = SmallGridworldMDP()
#             pi, v = policy_iteration(env, gamma=0.91)
#             return v, env
#     class ValueIteartionItem(GridworldDPItem):
#         title = "value iteration"
#
#         def get_value_function(self):
#             from irlc.ex09.value_iteration import value_iteration
#             from irlc.ex09.small_gridworld import SmallGridworldMDP
#             env = SmallGridworldMDP()
#             policy, v = value_iteration(env, gamma=0.92, theta=1e-6)
#             return v, env

# class GamlerQuestion(QuestionGroup):
#     title = "Gamblers problem"
#     class GamlerItem(GridworldDPItem):
#         title = "Value-function test"
#         def get_value_function(self):
#             # from irlc.ex09.small_gridworld import SmallGridworldMDP, plot_value_function
#             # from irlc.ex09.policy_iteration import policy_iteration
#             from irlc.ex09.value_iteration import value_iteration
#             from irlc.ex09.gambler import GamblerEnv
#             env = GamblerEnv()
#             pi, v = value_iteration(env, gamma=0.91)
#             return v, env

# class JackQuestion(QuestionGroup):
#     title ="Jacks car rental problem"
#     class JackItem(GridworldDPItem):
#         title = "Value function test"
#         max_cars = 5
#         tol = 0.01
#         def get_value_function(self):
#             from irlc.ex09.value_iteration import value_iteration
#             from irlc.ex09.jacks_car_rental import JackRentalMDP
#             env = JackRentalMDP(max_cars=self.max_cars, verbose=True)
#             pi, V = value_iteration(env, gamma=.9, theta=1e-3, max_iters=1000, verbose=True)
#             return V, env

class Problem8ValueIterationAgent(UTestCase):
    """ Value-iteration agent test """

    def test_sutton_gridworld(self):
        tol = 1e-2
        from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment
        env = SuttonCornerGridEnvironment(living_reward=-1)
        from irlc.ex08.value_iteration_agent import ValueIterationAgent
        agent = ValueIterationAgent(env, mdp=env.mdp)
        stats, _ = train(env, agent, num_episodes=1000)
        self.assertL2(np.mean([s['Accumulated Reward'] for s in stats]), tol=tol)

    def test_bookgrid_gridworld(self):
        tol = 1e-2
        from irlc.gridworld.gridworld_environments import BookGridEnvironment
        env = BookGridEnvironment(living_reward=-1)
        from irlc.ex08.value_iteration_agent import ValueIterationAgent
        agent = ValueIterationAgent(env, mdp=env.mdp)
        stats, _ = train(env, agent, num_episodes=1000)
        self.assertL2(np.mean([s['Accumulated Reward'] for s in stats]), tol=tol)


    #
    #
    #     pass
    # class ValueAgentItem(GridworldDPItem):
    #     title = "Evaluation on Suttons small gridworld"
    #     tol = 1e-2
    #     def get_env(self):
    #         from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment
    #         return SuttonCornerGridEnvironment(living_reward=-1)
    #
    #     def compute_answer_print(self):
    #         env = self.get_env()
    #         from irlc.ex09.value_iteration_agent import ValueIterationAgent
    #         agent = ValueIterationAgent(env, mdp=env.mdp)
    #         # env = VideoMonitor(env, agent=agent, agent_monitor_keys=('v',))
    #         stats, _ = train(env, agent, num_episodes=1000)
    #         return np.mean( [s['Accumulated Reward'] for s in stats])
    #
    #     def process_output(self, res, txt, numbers):
    #         return res

    # class BookItem(ValueAgentItem):
    #     title = "Evaluation on alternative gridworld (Bookgrid)"
    #     def get_env(self):
    #         from irlc.gridworld.gridworld_environments import BookGridEnvironment
    #         return BookGridEnvironment(living_reward=-0.6)

# class DPAgentRLQuestion(QuestionGroup):
#     title = "Value-iteration agent test"
#     class ValueAgentItem(GridworldDPItem):
#         title = "Evaluation on Suttons small gridworld"
#         tol = 1e-2
#         def get_env(self):
#             from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment
#             return SuttonCornerGridEnvironment(living_reward=-1)
#
#         def compute_answer_print(self):
#             env = self.get_env()
#             from irlc.ex09.value_iteration_agent import ValueIterationAgent
#             agent = ValueIterationAgent(env, mdp=env.mdp)
#             # env = VideoMonitor(env, agent=agent, agent_monitor_keys=('v',))
#             stats, _ = train(env, agent, num_episodes=1000)
#             return np.mean( [s['Accumulated Reward'] for s in stats])
#
#         def process_output(self, res, txt, numbers):
#             return res
#
#     class BookItem(ValueAgentItem):
#         title = "Evaluation on alternative gridworld (Bookgrid)"
#         def get_env(self):
#             from irlc.gridworld.gridworld_environments import BookGridEnvironment
#             return BookGridEnvironment(living_reward=-0.6)

class Week08Tests(Report):
    title = "Tests for week 08"
    pack_imports = [irlc]
    individual_imports = []
    questions = [ (Problem1_to_3_Warmup, 10),
                  (Problem4PolicyEvaluation, 10),
                  (Problem5PolicyIteration, 10),
                  (Problem6ValueIteration, 10),
                  (Problem8ValueIterationAgent, 10),
                  (Problem9Gambler, 10),
                  ]
    # (JackQuestion, 10),
    # (ValueFunctionTest, 20),


if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week08Tests())
