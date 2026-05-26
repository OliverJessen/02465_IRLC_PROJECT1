from unitgrade import UTestCase, Report, hide

class DP1(UTestCase):
    def test_a_test_expected_items_next_day(self):
        from irlc.exam.midterm2023a.question_dp import a_expected_items_next_day
        self.assertAlmostEqual(a_expected_items_next_day(x=0, u=1), 0.1, places=5)

    def test_b_test_expected_items_next_day(self):
        from irlc.exam.midterm2023a.question_dp import b_evaluate_policy
        pi = self.get_pi()
        self.assertAlmostEqual(b_evaluate_policy(pi, 1), 2.7, places=5)

    def get_pi(self):
        from irlc.exam.midterm2023a.inventory import InventoryDPModel
        model = InventoryDPModel()
        pi = [{x: 1 if x == 0 else 0 for x in model.S(k)} for k in range(model.N)]
        return pi


class QuestionPID(UTestCase):
    def test_a(self):
        from irlc.exam.midterm2023a.question_pid import a_pid_Kp
        xs, xstar = self.get_problem()
        self.assertAlmostEqual(a_pid_Kp(xs, xstar=0, Kp=.5), -1, places=5)

    def test_b(self):
        from irlc.exam.midterm2023a.question_pid import b_pid_full
        xs, xstar = self.get_problem()
        self.assertAlmostEqual(b_pid_full(xs, xstar, Kp=.5, Ki=0.05, Kd=0.25), -4.2, places=5)

    def test_c(self):
        from irlc.exam.midterm2023a.question_pid import c_pid_stable
        xs, xstar = self.get_problem()
        self.assertAlmostEqual(c_pid_stable(xs, xstar, Kp=.5, Ki=0.05, Kd=0.25), -4.075, places=5)

    def get_problem(self):
        return  [10, 8, 7, 5, 3, 1, 0, -2, -1, 0, 2] , -1


class Midterm2023A(Report):
    title = "02465: Midterm A"
    import irlc
    pack_imports = [irlc]
    abbreviate_questions = True

    q1_questions = [
                    (DP1, 10),
                    (QuestionPID, 10)
                     ]

    questions = []
    questions += q1_questions


if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Midterm2023A())
