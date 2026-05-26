from unitgrade import UTestCase, Report, hide
import numpy as np
from irlc.exam.midterm2023b.question_mdp import a_get_reward, b_get_best_immediate_action, c_get_best_action_twosteps, SmallGambler
from irlc.exam.midterm2023b.question_td0 import a_compute_deltas, b_perform_td0, c_perform_td0_batched

class QuestionMDP(UTestCase):
    def get_sa(self):
        s = 16
        a = 26
        return s, a
    def test_a_test_expected_items_next_day(self):
        s,a = self.get_sa()
        self.assertEqualC(a_get_reward(s, a))

    def test_b_test_expected_items_next_day(self):
        s, a = self.get_sa()
        self.assertEqualC(b_get_best_immediate_action(s))

    def test_c_test_expected_items_next_day(self):
        s, a = self.get_sa()
        self.assertEqualC(c_get_best_action_twosteps(s))

class QuestionTD0(UTestCase):

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



class Midterm2023B(Report):
    title = "02465: Midterm B"
    import irlc
    pack_imports = [irlc]
    abbreviate_questions = True

    q1_questions = [
                    (QuestionMDP, 10),
                    (QuestionTD0, 10)
                     ]

    questions = []
    questions += q1_questions


if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Midterm2023B())
