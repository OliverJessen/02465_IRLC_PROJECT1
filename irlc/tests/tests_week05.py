# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex05.model_boeing import BoeingEnvironment
from unitgrade import UTestCase, Report
import irlc
from irlc import train
import numpy as np
from irlc.ex04.locomotive import LocomotiveEnvironment
from irlc.ex04.model_harmonic import HarmonicOscilatorEnvironment

matrices = ['L', 'l', 'V', 'v', 'vc']

class Problem3LQR(UTestCase):
    title = "LQR, full check of implementation"

    @classmethod
    def setUpClass(cls):
        # def init(self):
        from irlc.ex05.dlqr_check import check_LQR
        (cls.L, cls.l), (cls.V, cls.v, cls.vc) = check_LQR()
        # self.M = list(zip(matrices, [L, l, V, v, vc]))

    def chk_item(self, m_list):
        self.assertIsInstance(m_list, list)
        self.assertEqualC(len(m_list))
        for m in m_list:
            self.assertIsInstance(m, np.ndarray)
            self.assertEqualC(m.shape)
            self.assertL2(m, tol=1e-6)

    def test_L(self):
        self.chk_item(self.__class__.L)

    def test_l(self):
        self.chk_item(self.__class__.l)

    def test_V(self):
        self.chk_item(self.__class__.V)

    def test_v(self):
        self.chk_item(self.__class__.v)

    def test_vc(self):
        vc = self.__class__.vc
        self.assertIsInstance(vc, list)
        for d in vc:
            self.assertL2(d, tol=1e-6)

        self.chk_item(self.__class__.l)

class Problem4LQRAgent(UTestCase):
    def _mkagent(self, val=0.):
        A = np.ones((2, 2))* (1+val)
        A[1, 0] = 0
        B = np.asarray([[0], [1]])
        Q = np.eye(2) * (3+val)
        R = np.ones((1, 1)) * 2
        q = np.asarray([-1.1 + val, 0])
        from irlc.ex05.lqr_agent import LQRAgent
        env = LocomotiveEnvironment(render_mode=None, Tmax=5, slope=1)
        agent = LQRAgent(env, A=A, B=B, Q=Q, R=R, q=q)
        return agent

    def test_policy_lqr_a(self):
        agent = self._mkagent(0)
        self.assertL2(agent.pi(np.asarray([1, 0]), k=0))
        self.assertL2(agent.pi(np.asarray([1, 0]), k=5))

    def test_policy_lqr_b(self):
        agent = self._mkagent(0.2)
        self.assertL2(agent.pi(np.asarray([1, 0]), k=0))
        self.assertL2(agent.pi(np.asarray([1, 0]), k=5))

class Problem5_6_Boeing(UTestCase):

    def test_compute_A_B_d(self):
        from irlc.ex05.boeing_lqr import compute_A_B_d, compute_Q_R_q
        model = BoeingEnvironment(Tmax=10).discrete_model.continuous_model
        A, B, d = compute_A_B_d(model, dt=0.2)
        self.assertL2(A)
        self.assertL2(B)
        self.assertL2(d)

    def test_compute_Q_R_q(self):
        from irlc.ex05.boeing_lqr import compute_A_B_d, compute_Q_R_q
        model = BoeingEnvironment(Tmax=10).discrete_model.continuous_model
        Q, R, q = compute_Q_R_q(model, dt=0.2)
        self.assertL2(Q)
        self.assertL2(R)
        self.assertL2(q)

    def test_boing_path(self):
        from irlc.ex05.boeing_lqr import boeing_simulation
        stats, trajectories, env = boeing_simulation()
        self.assertL2(trajectories[-1].state, tol=1e-6)


class Problem7_8_PidLQR(UTestCase):
    def test_constant_lqr_agent(self):
        Delta = 0.06  # Time discretization constant
        # Define a harmonic osscilator environment. Use .., render_mode='human' to see a visualization.
        env = HarmonicOscilatorEnvironment(Tmax=8, dt=Delta, m=0.5, R=np.eye(1) * 8,
                                           render_mode=None)  # set render_mode='human' to see the oscillator.
        model = env.discrete_model.continuous_model  # Get the ControlModel corresponding to this environment.

        from irlc.ex05.boeing_lqr import compute_A_B_d, compute_Q_R_q
        from irlc.ex05.lqr_pid import ConstantLQRAgent
        A, B, d = compute_A_B_d(model, Delta)
        Q, R, q = compute_Q_R_q(model, Delta)
        x0, _ = env.reset()

        # Run the LQR agent
        lqr_agent = ConstantLQRAgent(env, A=A, B=B, d=d, Q=Q, R=R, q=q)
        self.assertLinf(lqr_agent.pi(x0, k=0), tol=1e-3)
        self.assertLinf(lqr_agent.pi(x0, k=10), tol=1e-3)


    def test_KpKd(self):
        Delta = 0.06  # Time discretization constant
        # Define a harmonic osscilator environment. Use .., render_mode='human' to see a visualization.
        env = HarmonicOscilatorEnvironment(Tmax=8, dt=Delta, m=0.5, R=np.eye(1) * 8,
                                           render_mode=None)  # set render_mode='human' to see the oscillator.
        model = env.discrete_model.continuous_model  # Get the ControlModel corresponding to this environment.
        from irlc.ex05.boeing_lqr import compute_A_B_d, compute_Q_R_q
        from irlc.ex05.lqr_pid import ConstantLQRAgent, get_Kp_Kd
        A, B, d = compute_A_B_d(model, Delta)
        Q, R, q = compute_Q_R_q(model, Delta)
        lqr_agent = ConstantLQRAgent(env, A=A, B=B, d=d, Q=Q, R=R, q=q)
        Kp, Kd = get_Kp_Kd(lqr_agent.L[0])
        self.assertAlmostEqualC(Kp, places=3)
        self.assertAlmostEqualC(Kd, places=3)




class Week05Tests(Report):
    title = "Tests for week 05"
    pack_imports = [irlc]
    individual_imports = []
    questions = [
        (Problem3LQR, 10),
        (Problem4LQRAgent, 10),
        (Problem5_6_Boeing, 10),
        (Problem7_8_PidLQR, 10),
                 ]
if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week05Tests())
