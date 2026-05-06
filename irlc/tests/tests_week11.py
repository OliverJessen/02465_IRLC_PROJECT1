# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import UTestCase, Report, cache
from irlc import train
import irlc.ex09.envs
import gymnasium as gym
from irlc.tests.tests_week07 import train_recording
from irlc.tests.tests_week10 import TabularAgentStub

# This problem no longer exists.
# class NStepSarseEvaluationQuestion(TD0Question):
#     """ Test of TD-n evaluation agent """
#     # class EvaluateTabular(VExperienceItem):
#     #     title = "Value-function test"
#     gamma = 0.8
#     def get_env_agent(self):
#         envn = "SmallGridworld-v0"
#         from irlc.ex11.nstep_td_evaluate import TDnValueAgent
#         env = gym.make(envn)
#         agent = TDnValueAgent(env, gamma=self.gamma, n=5)
#         return env, agent




class NStepSarsaQuestion(TabularAgentStub):
    title = "N-step Sarsa"
    # class SarsaReturnItem(SarsaQuestion):
    def get_env_agent(self):
        from irlc.ex11.nstep_sarsa_agent import SarsaNAgent
        agent = SarsaNAgent(self.get_env(), gamma=self.gamma, n=5)
        return agent.env, agent

    def test_accumulated_reward(self):
        self.tol_qs = 2.7
        self.chk_accumulated_reward()


class LinearAgentStub(UTestCase):
    # class LinearExperienceItem(LinearWeightVectorTest):
    tol = 1e-6
    # title = "Linear sarsa agent"
    alpha = 0.08
    num_episodes = 300
    # title = "Weight-vector test"
    # testfun = QPrintItem.assertL2
    gamma = 0.8
    tol_w = 1e-5


    def get_env_agent(self):
        raise NotImplementedError()

    def get_env(self):
        return gym.make("MountainCar500-v0")

    # def get_env_agent(self):
    #     return None, None

    @cache
    def compute_trajectories(self):
        env, agent = self.get_env_agent()
        _, trajectories = train(env, agent, return_trajectory=True, num_episodes=1, max_steps=100)
        return trajectories, agent.Q.w

    def chk_Q_weight_vector_w(self):
        trajectories, w = self.compute_trajectories()
        env, agent = self.get_env_agent()
        train_recording(env, agent, trajectories)
        print(w)
        print(agent.Q.w)
        self.assertL2(agent.Q.w, w, tol=self.tol_w)

    pass
class LinearSarsaAgentQuestion(LinearAgentStub):
    """ Sarsa Agent with linear function approximators """

    def get_env_agent(self):
        env = self.get_env()
        from irlc.ex11.semi_grad_sarsa import LinearSemiGradSarsa
        agent = LinearSemiGradSarsa(env, gamma=1, alpha=self.alpha, epsilon=0)
        return env, agent

    def test_Q_weight_vector_w(self):
        self.tol_w = 1.4
        self.chk_Q_weight_vector_w()

class LinearQAgentQuestion(LinearAgentStub):
    """ Test of Linear Q Agent """

    def get_env_agent(self):
        env = self.get_env()
        alpha = 0.1
        from irlc.ex11.semi_grad_q import LinearSemiGradQAgent
        agent = LinearSemiGradQAgent(env, gamma=1, alpha=alpha, epsilon=0)
        return env, agent

    def test_Q_weight_vector_w(self):
        # self.tol_qs = 1.9
        self.tol_w = 7
        self.chk_Q_weight_vector_w()


# class SimpleQLearningQuestion(UTestCase):
#     def test_a_greedy_policy(self):
#         from irlc.ex11.old.question_simple_q import a_greedy_policy
#
#         states = [0, 1, 2]
#         actions = [0, 1]
#         q_example = {}  # Initialize a small example of Q-values.
#         for s in states:
#             for a in actions:
#                 q_example[s, a] = s / 2 + 2 ** a  # Initialize so that Q(s, a) = s / 2 + 2**a
#         self.assertEqual(a_greedy_policy(q_example, state=0), 1)
#
#
#     def test_b_update_single_q(self):
#         from irlc.ex11.old.question_simple_q import b_update_single_q
#         states = [0, 1, 2]
#         actions = [0, 1]
#         q_example = {}  # Initialize a small example of Q-values.
#         for s in states:
#             for a in actions:
#                 q_example[s, a] = s / 2 + 2 ** a  # Initialize so that Q(s, a) = s / 2 + 2**a
#
#         alpha = 0.8
#         gamma = 0.9
#         state = 0
#         action = 1
#         reward = 0.8
#         next_state = 2
#         self.assertAlmostEqual(b_update_single_q(alpha, gamma, q_example, state, action, reward, next_state), 3.2, places=4)
#
#     def test_c_update_all_q(self):
#         from irlc.ex11.old.question_simple_q import c_update_all_q
#
#         alpha = 0.8
#         gamma = 0.9
#         state = 0
#         action = 1
#
#         # The trajectory is of the form [..., (S_t, A_t, R_{t+1}), ... ]
#         example_trajectory = [(0, 1, 0.5),
#                               (2, 0, -0.75),
#                               (0, 1, 0.5),
#                               (1, 0, 0.5)]
#
#         updated_q_values = c_update_all_q(alpha, gamma, example_trajectory)  # This should be a dictionary.
#         self.assertAlmostEqual(updated_q_values[state, action], 0.48, places=4)
#
#     def test_c_update_all_q_b(self):
#         from irlc.ex11.old.question_simple_q import c_update_all_q
#         example_trajectory = [(0, 1, 0.5),
#                               (2, 0, -0.75),
#                               (0, 1, 0.5),
#                               (1, 0, 0.5)]
#
#         updated_q_values = c_update_all_q(alpha=0.8, gamma=0.9, states_actions_rewards=example_trajectory)
#         for (s,a, _) in example_trajectory[:-2]:
#             self.assertAlmostEqualC(updated_q_values[s, a], places=4)


class Week11Tests(Report):
    title = "Tests for week 11"
    pack_imports = [irlc]
    individual_imports = []
    questions =[
        # (NStepSarseEvaluationQuestion, 10),
        (LinearQAgentQuestion, 10),
        (LinearSarsaAgentQuestion, 10),
        # (SimpleQLearningQuestion, 10),
        (NStepSarsaQuestion, 5),
        ]
if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week11Tests())
