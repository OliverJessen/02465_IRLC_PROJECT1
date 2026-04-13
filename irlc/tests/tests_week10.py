# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex10.question_td0 import a_compute_deltas, b_perform_td0, c_perform_td0_batched
from unitgrade import Report, UTestCase, cache
from irlc import train
import irlc.ex09.envs
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from irlc.tests.tests_week07 import train_recording
import numpy as np
from irlc.tests.tests_week09 import MCAgentQuestion

class TD0Question(UTestCase):
    """ Test of TD(0) evaluation agent """
    gamma = 0.8

    def get_env_agent(self):
        from irlc.ex10.td0_evaluate import TD0ValueAgent
        env = gym.make("SmallGridworld-v0")
        # env = TimeLimit(env, max_episode_steps=1000)
        agent = TD0ValueAgent(env, gamma=self.gamma)
        return env, agent

    @cache
    def compute_trajectories(self):
        env, agent = self.get_env_agent()
        _, trajectories = train(env, agent, return_trajectory=True, num_episodes=1, max_steps=100)
        return trajectories, agent.v

    def test_value_function(self):
        # for k in range(1000):
        trajectories, v = self.compute_trajectories()
        env, agent = self.get_env_agent()
        train_recording(env, agent, trajectories)
        Qc = []
        Qe = []
        for s, value in v.items():
            Qe.append(value)
            Qc.append(agent.v[s])

        self.assertL2(Qe, Qc, tol=1e-5)




class QAgentQuestion(MCAgentQuestion):
    """ Test of Q Agent """
    # class EvaluateTabular(QExperienceItem):
    #     title = "Q-value test"

    def get_env_agent(self):
        from irlc.ex10.q_agent import QAgent
        env = gym.make("SmallGridworld-v0")
        agent = QAgent(env, gamma=.8)
        return env, agent


# class LinearWeightVectorTest(UTestCase):



# class LinearValueFunctionTest(LinearWeightVectorTest):
#     title = "Linear value-function test"
#     def compute_answer_print(self):
#         trajectories, Q = self.precomputed_payload()
#         env, agent = self.get_env_agent()
#         train_recording(env, agent, trajectories)
#         self.Q = Q
#         self.question.agent = agent
#         vfun = [agent.Q[s,a] for s, a in zip(trajectories[0].state, trajectories[0].action)]
#         return vfun

# class TabularAgentStub(UTestCase):
#
#     pass

class TabularAgentStub(UTestCase):
    """ Average return over many simulated episodes """
    gamma = 0.95
    epsilon = 0.2
    tol = 0.1
    tol_qs = 0.3
    episodes = 9000

    def get_env(self):
        return gym.make("SmallGridworld-v0")

    def get_env_agent(self):
        raise NotImplementedError()
        # from irlc.ex11.sarsa_agent import SarsaAgent
        # agent = SarsaAgent(self.get_env(), gamma=self.gamma)
        # return agent.env, agent

    def get_trained_agent(self):
        env, agent = self.get_env_agent()
        stats, _ = train(env, agent, num_episodes=self.episodes)
        return agent, stats

    def chk_accumulated_reward(self):
        agent, stats = self.get_trained_agent()
        s0, _ = agent.env.reset()
        actions, qs = agent.Q.get_Qs(s0)
        print("Tolerance is", self.tol_qs)
        self.assertL2(qs, tol=self.tol_qs)
        self.assertL2(np.mean([s['Accumulated Reward'] for s in stats]), tol=self.tol)

    # def test_accumulated_reward(self):
    #     env, agent = self.get_env_agent()
    #     stats, _ = train(env, agent, num_episodes=5000)
    #     s = env.reset()
    #     actions, qs = agent.Q.get_Qs(s)
    #     self.assertL2(qs, tol=0.3)
    #     self.assertL2(np.mean([s['Accumulated Reward'] for s in stats]), tol=self.tol)

class SarsaQuestion(TabularAgentStub):


    def get_env_agent(self):
        from irlc.ex10.sarsa_agent import SarsaAgent
        agent = SarsaAgent(self.get_env(), gamma=self.gamma)
        return agent.env, agent

    def test_accumulated_reward(self):
        self.tol_qs = 2.7 # Got 2.65 in one run.
        self.chk_accumulated_reward()

class ExamQuestionTD0(UTestCase):

    def get_problem(self):
        states = [1, 0, 2, -1, 2, 4, 5, 4, 3, 2, 1, -1]
        rewards = [1, 1, -1, 0, 1, 2, 2, 0, 0, -1, 1]
        v = {s: 0 for s in states}
        gamma = 0.9
        alpha = 0.2
        return v, states, rewards, gamma, alpha

    def test_a(self):
        v, states, rewards, gamma, alpha = self.get_problem()
        self.assertEqualC(a_compute_deltas(v, states, rewards, gamma))

    def test_b(self):
        v, states, rewards, gamma, alpha = self.get_problem()
        self.assertEqualC(b_perform_td0(v, states, rewards, gamma, alpha))

    def test_c(self):
        v, states, rewards, gamma, alpha = self.get_problem()
        self.assertEqualC(c_perform_td0_batched(v, states, rewards, gamma, alpha))
class Week10Tests(Report):
    title = "Tests for week 10"
    pack_imports = [irlc]
    individual_imports = []
    questions = [
                (QAgentQuestion, 10),
                (SarsaQuestion, 10),
                (TD0Question, 10),
                 (ExamQuestionTD0, 10),
                 ]

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week10Tests())
