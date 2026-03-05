# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
# from irlc.ex02.graph_traversal import pi_inc, pi_smart, pi_silly, policy_rollout, SmallGraphDP
from collections import defaultdict
from unitgrade import Report
import irlc
from unitgrade import UTestCase
from irlc.ex02.inventory import InventoryDPModel, DP_stochastic
from irlc.ex02.deterministic_inventory import DeterministicInventoryDPModel

def gN_dp(self, env):
    for s in sorted(self.env.S(self.env.N)):
        self.assertLinf(self.env.gN(s))

def f_dp(self, env):
    self.assertEqualC(self.env.N)

    for k in range(self.env.N):
        for s in sorted(self.env.S(k)):
            for a in sorted(self.env.A(s,k)):
                from collections import defaultdict

                dd_f = defaultdict(float)
                # dd_g = defaultdict(float)

                for w, pw in self.env.Pw(s,a,k).items():
                    dd_f[(s,a, self.env.f(s,a,w,k))] += pw
                    # dd_g[(s, a, self.env.g(s, a, w, k))] += pw

                # Check transition probabilities sum to 1.
                self.assertAlmostEqual(sum(dd_f.values()), 1, places=6)
                # self.assertAlmostEqual(sum(dd_g.values()), 1, places=6)

                for key in sorted(dd_f.keys()):
                    self.assertEqualC(key)
                    self.assertLinf(dd_f[key], tol=1e-7)

                # for key in sorted(dd_g.keys()):
                #     self.assertEqualC(key)
                #     self.assertLinf(dd_g[key], tol=1e-7)

def g_dp(self, env):
    for k in range(self.env.N):
        for s in sorted(self.env.S(k)):
            for a in sorted(self.env.A(s, k)):

                # dd_f = defaultdict(float)
                dd_g = defaultdict(float)

                for w, pw in self.env.Pw(s, a, k).items():
                    # dd_f[(s, a, self.env.f(s, a, w, k))] += pw
                    dd_g[(s, a, self.env.g(s, a, w, k))] += pw

                # Check transition probabilities sum to 1.
                # self.assertAlmostEqual(sum(dd_f.values()), 1, places=6)
                self.assertAlmostEqual(sum(dd_g.values()), 1, places=6)

                # for key in sorted(dd_f.keys()):
                #     self.assertEqualC(key)
                #     self.assertLinf(dd_f[key], tol=1e-7)

                for key in sorted(dd_g.keys()):
                    self.assertEqualC(key)
                    self.assertLinf(dd_g[key], tol=1e-7)




class Problem2DeterministicInventory(UTestCase):
    @property
    def model(self):
        return DeterministicInventoryDPModel()

    def test_Pw(self):
        for k in range(self.model.N):
            self.assertEqual(self.model.Pw(0, 1, k), {k+1: 1} )



    # def test_f(self):
    #     f_dp(self, self.env)

    # def test_g(self):
    #     g_dp(self, self.env)
    #
    # def test_gN(self):
    #     gN_dp(self, self.env)


class Problem3StochasticDP(UTestCase):
    """ Inventory control """
    def test_policy(self):
        inv = InventoryDPModel()
        J, pi = DP_stochastic(inv)

        # Test action at time step N-1
        self.assertEqual(pi[-1][0], 1)
        self.assertEqual(pi[-1][1], 0)
        self.assertEqual(pi[-1][2], 0)

        # test all actions at time step N-1
        self.assertEqualC(pi[-1])

        # Test all actions at all time steps
        self.assertEqualC(pi)

    def test_J(self):
        inv = InventoryDPModel()
        J, pi = DP_stochastic(inv)

        self.assertLinf(J[-1][0], tol=1e-8)
        self.assertLinf(J[-1][1], tol=1e-8)
        self.assertLinf(J[-1][2], tol=1e-8)
        
        for k in range(len(J)):
            for x in [0,1,2]:
                print("testing", J[k][x])
                self.assertLinf(J[k][x], tol=1e-8)

class Problem4DPAgent(UTestCase):
    def test_agent(self):
        from irlc.ex01.inventory_environment import InventoryEnvironment
        from irlc.ex02.inventory import InventoryDPModel
        from irlc.ex02.dp_agent import DynamicalProgrammingAgent
        env = InventoryEnvironment(N=3)
        inventory = InventoryDPModel(N=3)
        agent = DynamicalProgrammingAgent(env, model=inventory)
        s0, _ = env.reset()
        self.assertEqualC(agent.pi(s0, 0)) # We just test the first action.


# class DPChessMatch(UTestCase):
#     """ Chessmatch """
#     def test_J(self):
#         N = 2
#         pw = 0.45
#         pd = 0.8
#         cm = ChessMatch(N, pw=pw, pd=pd)
#         J, pi = DP_stochastic(cm)
#         self.assertLinf(J[-1][0], tol=1e-4)
#         self.assertLinf(J[-2][0], tol=1e-4)
#         self.assertLinf(J[0][0], tol=1e-4)




# class SmallGraphPolicies(UTestCase):
#     """ Test the policies in the small graph environment """
#     def test_pi_smart(self):
#         self.assertEqual(pi_smart(1, 0), 5)
#
#     def test_pi_inc(self):
#         from irlc.ex02.graph_traversal import pi_inc, pi_smart, pi_silly
#         for k in range(5):
#             self.assertEqual(pi_inc(k+1, k), k+2)
#             # self.assertEqual(pi_smart(k + 1, k), 5)
#             # self.assertEqual(pi_smart(k + 1, k), 5)
#
#     def test_rollout(self):
#         # self.assertEqual(3, 1)
#         t = 5
#         x0 = 1  # starting node
#         model = SmallGraphDP(t=t)
#
#         self.assertEqualC(policy_rollout(model, pi_silly, x0)[0])
#         self.assertEqualC(policy_rollout(model, pi_smart, x0)[0])
#         self.assertEqualC(policy_rollout(model, pi_inc, x0)[0])

class Problem2DeterministicDP(UTestCase):
    def test_dp_deterministic(self):
        model = DeterministicInventoryDPModel()
        J, pi = DP_stochastic(model)

        self.assertLinf(J[-1][0], tol=1e-5)
        self.assertLinf(J[-1][1], tol=1e-5)
        self.assertLinf(J[-1][2], tol=1e-5)

        self.assertLinf(J[0][0], tol=1e-5)
        self.assertLinf(J[0][1], tol=1e-5)
        self.assertLinf(J[0][2], tol=1e-5)



# class TestInventoryModel(UTestCase):
#     @property
#     def env(self):
#         return InventoryDPModel()
#
#     def test_gN(self):
#         gN_dp(self, self.env)
#
#     def test_f(self):
#         f_dp(self, self.env)
#
#     def test_g(self):
#         g_dp(self, self.env)



# class TestFrozenDP(UTestCase):
#     @property
#     def env(self):
#         return Gym2DPModel(gym_env=gym.make("FrozenLake-v1"))
#
#     def test_f(self):
#         f_dp(self, self.env)
#
#     def test_g(self):
#         g_dp(self, self.env)

class ExamQuestion7FlowersStore(UTestCase):
    def test_a_get_policy(self):
        from irlc.ex02.flower_store import a_get_policy
        x0 = 0
        c = 0.5
        N = 3
        self.assertEqual(a_get_policy(N, c, x0), 1)

    def test_b_prob_empty(self):
        from irlc.ex02.flower_store import b_prob_one
        x0 = 0
        N = 3
        self.assertAlmostEqual(b_prob_one(N, x0), 0.492, places=2)


class Week02Tests(Report):
    title = "Tests for week 02"
    pack_imports = [irlc]
    individual_imports = []
    questions = [
        (Problem2DeterministicInventory, 10),
        (Problem2DeterministicDP, 10),
        (Problem3StochasticDP, 10),
        (Problem4DPAgent, 10),
        (ExamQuestion7FlowersStore, 10),
         ]


# (SmallGraphPolicies, 10),
# (TestInventoryModel, 10),
# (DPChessMatch, 10),
# (TestFrozenDP, 10),

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week02Tests() )
