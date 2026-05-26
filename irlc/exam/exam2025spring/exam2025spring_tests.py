from unitgrade import UTestCase, Report, hide
import time
import numpy as np


class CakeInventoryQuestion(UTestCase):
    def test_a_expected_cost(self):
        from irlc.exam.exam2025spring.question_inventory_cake import a_expected_cost
        self.assertAlmostEqual(a_expected_cost(x0=0, u0=1), 1.3, places=4)







    # problem b
    def test_b_best_action(self):
        from irlc.exam.exam2025spring.question_inventory_cake import b_best_action
        self.assertEqual(b_best_action(N=3, cost_per_cake=0.8, k=0, x=1), 1)








    def test_c_lazy_baker(self):
        from irlc.exam.exam2025spring.question_inventory_cake import c_lazy_baker
        t0 = time.time()
        self.assertAlmostEqual(c_lazy_baker(N=3, cost_per_cake=0.7, x0=0), 1.311, places=4)
        time_taken = time.time()-t0
        self.assertLess(time_taken, 10, msg=f"The function takes too long to evaluate. It took {time_taken} seconds")













class SimpleQLearningQuestion(UTestCase):
    def test_a_greedy_policy(self):
        from irlc.exam.exam2025spring.question_simple_q import a_greedy_policy

        states = [0, 1, 2]
        actions = [0, 1]
        q_example = {}  # Initialize a small example of Q-values.
        for s in states:
            for a in actions:
                q_example[s, a] = s / 2 + 2 ** a  # Initialize so that Q(s, a) = s / 2 + 2**a
        self.assertEqual(a_greedy_policy(q_example, state=0), 1)







    def test_b_update_single_q(self):
        from irlc.exam.exam2025spring.question_simple_q import b_update_single_q
        states = [0, 1, 2]
        actions = [0, 1]
        q_example = {}  # Initialize a small example of Q-values.
        for s in states:
            for a in actions:
                q_example[s, a] = s / 2 + 2 ** a  # Initialize so that Q(s, a) = s / 2 + 2**a

        alpha = 0.8
        gamma = 0.9
        state = 0
        action = 1
        reward = 0.8
        next_state = 2
        self.assertAlmostEqual(b_update_single_q(alpha, gamma, q_example, state, action, reward, next_state), 3.2, places=4)







    def test_c_update_all_q(self):
        from irlc.exam.exam2025spring.question_simple_q import c_update_all_q

        alpha = 0.8
        gamma = 0.9
        state = 0
        action = 1

        # The trajectory is of the form [..., (S_t, A_t, R_{t+1}), ... ]
        example_trajectory = [(0, 1, 0.5),  # s_0 = 0, a_0 = 1, r_1 = 0.5  
                              (2, 0, -0.75),  # s_1 = 2, a_1 = 0, r_2 = -0.75
                              (0, 1, 0.5),  # s_2 = 0, a_2 = 1, r_3 = 0.4
                              (1, 0, 0.5)]  # s_3 = 1, a_3 = 0, r_4 = -0.75   

        updated_q_values = c_update_all_q(alpha, gamma, example_trajectory)  # This should be a dictionary.
        self.assertAlmostEqual(updated_q_values[state, action], 0.48, places=4)








class ControlPendulumQuestion(UTestCase):
    def test_a_dynamics_f(self):
        from irlc.exam.exam2025spring.question_control_pendulum import a_dynamics_f
        f_val = a_dynamics_f(theta=np.pi/2, thetadot=0, u=1)
        f1 = float(f_val[0])
        f2 = float(f_val[1])
        self.assertAlmostEqual(f1, 0, places=4)
        self.assertAlmostEqual(f2, 11.07, places=4)











    def test_b_euler(self):
        from irlc.exam.exam2025spring.question_control_pendulum import b_euler
        self.assertAlmostEqual(b_euler(theta0=0.1, thetadot0=0, delta=0.5, N=3), 0.8353, places=4)








class Exam2025Spring(Report):
    title = "02465 Exam Spring 2025"
    import irlc
    pack_imports = [irlc]
    abbreviate_questions = True

    questions = [
        (ControlPendulumQuestion, 12),
        (CakeInventoryQuestion, 12),
        (SimpleQLearningQuestion, 12)
    ]

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Exam2025Spring())
