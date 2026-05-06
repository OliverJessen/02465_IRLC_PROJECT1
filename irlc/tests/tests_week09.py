# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex10.question_td0 import a_compute_deltas, b_perform_td0, c_perform_td0_batched
from unitgrade import Report, UTestCase, cache
from irlc import train
import irlc.ex09.envs
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from irlc.tests.tests_week07 import train_recording


class MCAgentQuestion(UTestCase):
    """ Test of MC agent """
    def get_env_agent(self):
        from irlc.ex09.mc_agent import MCAgent
        env = gym.make("SmallGridworld-v0")
        env = TimeLimit(env, max_episode_steps=1000)
        gamma = .8
        agent = MCAgent(env, gamma=gamma, first_visit=True)
        return env, agent

    @cache
    def compute_trajectories(self):
        env, agent = self.get_env_agent()
        _, trajectories = train(env, agent, return_trajectory=True, num_episodes=1, max_steps=100)
        return trajectories, agent.Q.to_dict()

    def test_Q_function(self):
        trajectories, Q = self.compute_trajectories()
        env, agent = self.get_env_agent()
        train_recording(env, agent, trajectories)
        Qc = []
        Qe = []
        for s, qa in Q.items():
            for a,q in qa.items():
                Qe.append(q)
                Qc.append(agent.Q[s,a])

        self.assertL2(Qe, Qc, tol=1e-5)



class MCEvaluationQuestion(UTestCase):
    """ Test of MC evaluation agent """
    def get_env_agent(self):
        from irlc.ex09.mc_evaluate import MCEvaluationAgent
        env = gym.make("SmallGridworld-v0")
        env = TimeLimit(env, max_episode_steps=1000)
        gamma = .8
        agent = MCEvaluationAgent(env, gamma=gamma, first_visit=True)
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
class Week09Tests(Report):
    title = "Tests for week 9"
    pack_imports = [irlc]
    individual_imports = []
    questions = [(MCAgentQuestion, 10),
                (MCEvaluationQuestion, 10),
                 (ExamQuestionTD0, 10),
                 ]

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week09Tests())
