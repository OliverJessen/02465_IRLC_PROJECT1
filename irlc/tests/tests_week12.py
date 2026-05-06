# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import UTestCase, cache, Report
import irlc.ex09.envs
## WEEK 12:
from irlc.tests.tests_week11 import TabularAgentStub, LinearAgentStub

class LinearSarsaNstepAgentQuestion(LinearAgentStub):
    """ Test of Linear n-step sarsa Agent """
    tol = 2200
    num_episodes = 150
    gamma = 1
    tol_w = 2.5

    def get_env_agent(self):
        env = self.get_env()
        from irlc.ex12.semi_grad_nstep_sarsa import LinearSemiGradSarsaN
        from irlc.ex12.semi_grad_sarsa_lambda import alpha
        agent = LinearSemiGradSarsaN(env, gamma=self.gamma, alpha=alpha, epsilon=0)
        return env, agent

    def test_Q_weight_vector_w(self):

        self.chk_Q_weight_vector_w()


class LinearSarsaLambdaAgentQuestion(LinearAgentStub):
    """ Test of Linear sarsa(Lambda) Agent """
    tol = 2200
    num_episodes = 150
    gamma = 1
    tol_w = 15

    def get_env_agent(self):
        env = self.get_env()
        from irlc.ex12.semi_grad_sarsa_lambda import LinearSemiGradSarsaLambda, alpha
        agent = LinearSemiGradSarsaLambda(env, gamma=self.gamma, alpha=alpha, epsilon=0)
        return env, agent

    def test_Q_weight_vector_w(self):
        self.chk_Q_weight_vector_w()

class SarsaLambdaQuestion(TabularAgentStub):
    """ Sarsa(lambda) """
    def get_env_agent(self):
        from irlc.ex12.sarsa_lambda_agent import SarsaLambdaAgent
        agent = SarsaLambdaAgent(self.get_env(), gamma=self.gamma, lamb=0.7)
        return agent.env, agent

    def test_reward_function(self):
        self.tol_qs = 3.1
        self.chk_accumulated_reward()

class Week12Tests(Report):
    title = "Tests for week 12"
    pack_imports = [irlc]
    individual_imports = []
    questions = [
        (SarsaLambdaQuestion, 10),
        (LinearSarsaLambdaAgentQuestion, 10),
        (LinearSarsaNstepAgentQuestion, 10),]

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week12Tests())
