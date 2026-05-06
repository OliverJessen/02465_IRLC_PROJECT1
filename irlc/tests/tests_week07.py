# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import UTestCase, Report, cache
import numpy as np
from irlc import train


def train_recording(env, agent, trajectories):
    for t in trajectories:
        env.reset()
        for k in range(len(t.action)):
            s = t.state[k]
            r = t.reward[k]
            a = t.action[k]
            sp = t.state[k+1]
            agent.pi(s,k)
            agent.train(s, a, r, sp, done=k == len(t.action)-1)


class BanditQuestion(UTestCase):
    """ Value (Q) function estimate """
    tol = 1e-2 # tie-breaking in the gradient bandit is ill-defined.
    # testfun = QPrintItem.assertL2

    # def setUpClass(cls) -> None:
    #     from irlc.ex08.simple_agents import BasicAgent
    #     from irlc.ex08.bandits import StationaryBandit
    #     env = StationaryBandit(k=10, )
    #     agent = BasicAgent(env, epsilon=0.1)
    #     _, cls.trajectories = train(env, agent, return_trajectory=True, num_episodes=1, max_steps=100)
    #     cls.Q = agent.Q
    #     cls.env = env
    #     cls.agent = agent

    def get_env_agent(self):
        from irlc.ex07.simple_agents import BasicAgent
        from irlc.ex07.bandits import StationaryBandit
        env = StationaryBandit(k=10)
        agent = BasicAgent(env, epsilon=0.1)
        return env, agent

    @cache
    def get_trajectories(self):
        env, agent = self.get_env_agent()
        _, trajectories = train(env, agent, return_trajectory=True, num_episodes=1, max_steps=100)
        return trajectories

    # def precompute_payload(self):
    #     env, agent = self.get_env_agent()
    #     _, trajectories = train(env, agent, return_trajectory=True, num_episodes=1, max_steps=100)
    #     return trajectories, agent.Q


    def test_agent(self):
        trajectories = self.get_trajectories()
        env, agent = self.get_env_agent()
        train_recording(env, agent, trajectories)
        self.assertL2(agent.Q, tol=1e-5)
        # return agent.Q
        # self.Q = Q
        # self.question.agent = agent
        # return agent.Q

    # testfun = QPrintItem.assertL2

    def test_action_distributin(self):
        T = 10000
        tol = 1 / np.sqrt(T) * 5
        trajectories = self.get_trajectories()
        env, agent = self.get_env_agent()
        train_recording(env, agent, trajectories)
        # for k in self._cache.keys(): print(k)

        from collections import Counter
        counts = Counter([agent.pi(None, k) for k in range(T)])
        distrib = [counts[k] / T for k in range(env.k)]
        self.assertL2(np.asarray(distrib), tol=tol)


    # def process_output(self, res, txt, numbers):
    #     return res

    # def process_output(self, res, txt, numbers):
    #     return res
    #
    # def test(self, computed, expected):
    #     super().test(computed, self.Q)

# class BanditQuestion(QPrintItem):
#     # tol = 1e-6
#     tol = 1e-2 # tie-breaking in the gradient bandit is ill-defined.
#     title = "Value (Q) function estimate"
#     testfun = QPrintItem.assertL2
#
#     def get_env_agent(self):
#         from irlc.ex08.simple_agents import BasicAgent
#         from irlc.ex08.bandits import StationaryBandit
#         env = StationaryBandit(k=10, )
#         agent = BasicAgent(env, epsilon=0.1)
#         return env, agent
#
#     def precompute_payload(self):
#         env, agent = self.get_env_agent()
#         _, trajectories = train(env, agent, return_trajectory=True, num_episodes=1, max_steps=100)
#         return trajectories, agent.Q
#
#     def compute_answer_print(self):
#         trajectories, Q = self.precomputed_payload()
#         env, agent = self.get_env_agent()
#         train_recording(env, agent, trajectories)
#         self.Q = Q
#         self.question.agent = agent
#         return agent.Q
#
#     def process_output(self, res, txt, numbers):
#         return res
#
#     def test(self, computed, expected):
#         super().test(computed, self.Q)
#
# class BanditItemActionDistribution(QPrintItem):
#     # Assumes setup has already been done.
#     title = "Action distribution test"
#     T = 10000
#     tol = 1/np.sqrt(T)*5
#     testfun = QPrintItem.assertL2
#
#     def compute_answer_print(self):
#         # print("In agent print code")
#         from collections import Counter
#         counts = Counter( [self.question.agent.pi(None, k) for k in range(self.T)] )
#         distrib = [counts[k] / self.T for k in range(self.question.agent.env.k)]
#         return np.asarray(distrib)
#
#     def process_output(self, res, txt, numbers):
#         return res
#
# class BanditQuestion(QuestionGroup):
#     title = "Simple bandits"
#     class SimpleBanditItem(BanditItem):
#         #title = "Value function estimate"
#         def get_env_agent(self):
#             from irlc.ex08.simple_agents import BasicAgent
#             from irlc.ex08.bandits import StationaryBandit
#             env = StationaryBandit(k=10, )
#             agent = BasicAgent(env, epsilon=0.1)
#             return env, agent
#     class SimpleBanditActionDistribution(BanditItemActionDistribution):
#         pass



class GradientBanditQuestion(BanditQuestion):
    """ Gradient agent """
    # class SimpleBanditItem(BanditItem):
        # title = "Simple agent question"
    def get_env_agent(self):
        from irlc.ex07.bandits import StationaryBandit
        from irlc.ex07.gradient_agent import GradientAgent
        env = StationaryBandit(k=10)
        agent = GradientAgent(env, alpha=0.05)
        return env, agent

    # def precompute_payload(self):
    #     env, agent = self.get_env_agent()
    #     _, trajectories = train(env, agent, return_trajectory=True, num_episodes=1, max_steps=100)
    #     return trajectories

    def test_agent(self):
        trajectories = self.get_trajectories()
        env, agent = self.get_env_agent()
        train_recording(env, agent, trajectories)
        self.assertL2(agent.H, tol=1e-5)


    # def test(self, computed, expected):
    #     self.testfun(computed, self.H)
    #
    # class SimpleBanditActionDistribution(BanditItemActionDistribution):
    #     pass


# class GradientBanditQuestion(QuestionGroup):
#     title = "Gradient agent"
#     class SimpleBanditItem(BanditItem):
#         # title = "Simple agent question"
#         def get_env_agent(self):
#             from irlc.ex08.bandits import StationaryBandit
#             from irlc.ex08.gradient_agent import GradientAgent
#             env = StationaryBandit(k=10)
#             agent = GradientAgent(env, alpha=0.05)
#             return env, agent
#
#         def precompute_payload(self):
#             env, agent = self.get_env_agent()
#             _, trajectories = train(env, agent, return_trajectory=True, num_episodes=1, max_steps=100)
#             return trajectories, agent.H
#
#         def compute_answer_print(self):
#             trajectories, H = self.precomputed_payload()
#             env, agent = self.get_env_agent()
#             train_recording(env, agent, trajectories)
#             self.H = H
#             self.question.agent = agent
#             return agent.H
#
#         def test(self, computed, expected):
#             self.testfun(computed, self.H)
#
#     class SimpleBanditActionDistribution(BanditItemActionDistribution):
#         pass



class UCBAgentQuestion(BanditQuestion):
    """ UCB agent """
    # class UCBAgentItem(BanditItem):
    def get_env_agent(self):
        from irlc.ex07.bandits import StationaryBandit
        from irlc.ex07.ucb_agent import UCBAgent
        env = StationaryBandit(k=10)
        agent = UCBAgent(env)
        return env, agent

    # class UCBAgentActionDistribution(BanditItemActionDistribution):
    #     pass


# class UCBAgentQuestion(QuestionGroup):
#     title = "UCB agent"
#     class UCBAgentItem(BanditItem):
#         def get_env_agent(self):
#             from irlc.ex08.bandits import StationaryBandit
#             from irlc.ex08.ucb_agent import UCBAgent
#             env = StationaryBandit(k=10)
#             agent = UCBAgent(env)
#             return env, agent
#
#     class UCBAgentActionDistribution(BanditItemActionDistribution):
#         pass

# class NonstatiotnaryAgentQuestion(QuestionGroup):
#     title = "Nonstationary bandit environment"
#     class NonstationaryItem(BanditItem):
#         def get_env_agent(self):
#             epsilon = 0.1
#             from irlc.ex08.nonstationary import NonstationaryBandit, MovingAverageAgent
#             bandit = NonstationaryBandit(k=10)
#             agent = MovingAverageAgent(bandit, epsilon=epsilon, alpha=0.15)
#             return bandit, agent
#
# class NonstationaryActionDistribution(BanditItemActionDistribution):
#     pass

class NonstatiotnaryAgentQuestion(BanditQuestion):
    """ UCB agent """
    # class UCBAgentItem(BanditItem):
    def get_env_agent(self):
        epsilon = 0.1
        from irlc.ex07.nonstationary import NonstationaryBandit, MovingAverageAgent
        bandit = NonstationaryBandit(k=10)
        agent = MovingAverageAgent(bandit, epsilon=epsilon, alpha=0.15)
        return bandit, agent

import irlc
class Week07Tests(Report):
    title = "Tests for week 07"
    pack_imports = [irlc]
    individual_imports = []
    questions = [
            (BanditQuestion, 10),
            (GradientBanditQuestion, 10),
            (UCBAgentQuestion, 5),
            (NonstatiotnaryAgentQuestion, 5)
                ]

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week07Tests())
