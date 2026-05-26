from unitgrade import UTestCase, Report
import numpy as np
from unitgrade import hide

class QuestionBandit(UTestCase):
    def test_a_select_next_action_epsilon0(self):
        from irlc.exam.exam2023spring.question_bandit import a_select_next_action_epsilon0
        actions = [1, 0, 2, 1, 2, 4, 5, 4, 3, 2, 1, 1]
        rewards = [1, 1, 1, 0, 1, 3, 2, 0, 4, 1, 1, 2]
        K = 10
        a = a_select_next_action_epsilon0(K, actions, rewards)
        self.assertEqual(a, 3)




    def test_b_select_next_action(self):
        from irlc.exam.exam2023spring.question_bandit import b_select_next_action
        actions = [1, 0, 2, 1, 2, 4, 5, 4, 3, 2, 1, 1]
        rewards = [1, 1, 1, 0, 1, 3, 2, 0, 4, 1, 1, 2]
        K = 10
        a = b_select_next_action(K, actions, rewards, epsilon=0.3)
        self.assertIsInstance(int(a), int)
        self.assertTrue(0 <= a < K)





    def test_c_nonstationary_Qs(self):
        from irlc.exam.exam2023spring.question_bandit import c_nonstationary_Qs
        actions = [1, 0, 2, 1, 2, 4, 5, 4, 3, 2, 1, 1]
        rewards = [1, 1, 1, 0, 1, 3, 2, 0, 4, 1, 1, 2]
        K = 10
        Q = c_nonstationary_Qs(K, actions, rewards, alpha=0.1)
        self.assertL2(Q[5], 0.2, tol=0.001) # Test Q(5) = 0.2.






class QuestionInventory(UTestCase):
    def test_a_get_policy(self):
        from irlc.exam.exam2023spring.question_inventory import a_get_policy
        x0 = 0
        c = 0.5
        N = 3
        self.assertEqual(a_get_policy(N, c, x0), 1)





    def test_b_prob_empty(self):
        from irlc.exam.exam2023spring.question_inventory import b_prob_one
        x0 = 0
        N = 3
        self.assertAlmostEqual(b_prob_one(N, x0), 0.492, places=2)





class QuestionLQR(UTestCase):
    def test_a_LQR_solve(self):
        from irlc.exam.exam2023spring.question_lqr import a_LQR_solve
        a = 1
        x = np.asarray([1, 0])
        self.assertAlmostEqual(a_LQR_solve(a, x), -1.666, places=2)





    def test_b_linearize(self):
        from irlc.exam.exam2023spring.question_lqr import b_linearize
        theta = np.pi / 2
        A, B, d = b_linearize(theta)
        self.assertAlmostEqual(d[1],  4.91, places=2)







    def test_c_get_optimal_linear_policy(self):
        from irlc.exam.exam2023spring.question_lqr import c_get_optimal_linear_policy
        theta = 0.1
        self.assertAlmostEqual(c_get_optimal_linear_policy(x0=np.asarray([theta, 0])), -1.0716, places=3)
    





class Exam2023Spring(Report):
    title = "02465 Exam Spring 2023"
    import irlc
    pack_imports = [irlc]
    abbreviate_questions = True

    q1_questions = [
                    (QuestionInventory, 12),
                    (QuestionBandit, 12),
                    (QuestionLQR, 12)
                     ]

    questions = []
    questions += q1_questions


if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Exam2023Spring())
