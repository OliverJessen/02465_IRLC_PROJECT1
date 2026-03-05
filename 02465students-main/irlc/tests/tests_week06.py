# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import Report
import irlc
from unitgrade import UTestCase
import numpy as np
from irlc import Agent, train

class RendevouzItem(UTestCase):
    def test_rendevouz_without_linesearch(self):
        """ Rendevouz with iLQR (no linesearch) """
        from irlc.ex06.ilqr_rendovouz_basic import solve_rendovouz
        (xs, us, J_hist, l, L), env = solve_rendovouz(use_linesearch=False)
        # print(J_hist[-1])
        self.assertL2(xs[-1], tol=1e-2)

    def test_rendevouz_with_linesearch(self):
        """ Rendevouz with iLQR (with linesearch) """
        from irlc.ex06.ilqr_rendovouz_basic import solve_rendovouz
        (xs, us, J_hist, l, L), env = solve_rendovouz(use_linesearch=True)
        # print(J_hist[-1])
        self.assertL2(xs[-1], tol=1e-2)
        # return l, L, xs





class ILQRAgentQuestion(UTestCase):
    """ iLQR Agent on Rendevouz """
    def test_ilqr_agent(self):
        from irlc.ex06.ilqr_agent import solve_rendevouz
        stats, trajectories, agent = solve_rendevouz()
        self.assertL2(trajectories[-1].state[-1], tol=1e-2)


class ILQRPendulumQuestion(UTestCase):
    """ iLQR Agent on Pendulum """

    def test_ilqr_agent_pendulum(self):
        from irlc.ex06.ilqr_pendulum_agent import Tmax, N
        from irlc.ex04.model_pendulum import GymSinCosPendulumEnvironment
        from irlc.ex06.ilqr_agent import ILQRAgent
        dt = Tmax / N
        env = GymSinCosPendulumEnvironment(dt, Tmax=Tmax, supersample_trajectory=True)
        agent = ILQRAgent(env, env.discrete_model, N=N, ilqr_iterations=200, use_linesearch=True)
        stats, trajectories = train(env, agent, num_episodes=1, return_trajectory=True)
        state = trajectories[-1].state[-1]
        self.assertL2(state, tol=2e-2)

class Week07Tests(Report): #240 total.
    title = "Tests for week 07"
    pack_imports = [irlc]
    individual_imports = []
    questions = [
        (RendevouzItem, 10),  # ok
        (ILQRAgentQuestion, 10),  # ok
        (ILQRPendulumQuestion, 10),  # ok
                 ]

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week07Tests())
