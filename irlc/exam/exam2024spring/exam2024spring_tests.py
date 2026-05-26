from unitgrade import UTestCase, Report, hide

class InventoryBrideQuestion(UTestCase):
    def test_a_get_cost(self):
        from irlc.exam.exam2024spring.question_inventory import a_get_cost
        x0 = 0
        N = 6
        m = 4
        self.assertAlmostEqual(a_get_cost(N, m, x0), 13.75, places=3)






    def test_b_sale(self):
        from irlc.exam.exam2024spring.question_inventory import b_sale
        x0 = 0
        N = 6
        m = 4
        self.assertAlmostEqual(b_sale(N, m, x0), 11.25, places=3)






class MDPQuestion(UTestCase):
    def test_a_always_airbnb(self):
        from irlc.exam.exam2024spring.question_bill_mdp import a_always_airbnb
        self.assertAlmostEqual(a_always_airbnb(r_airbnb=0.01, gamma=0.99), 1, places=2)






    def test_b_random_decisions(self):
        from irlc.exam.exam2024spring.question_bill_mdp import b_random_decisions
        self.assertAlmostEqual(b_random_decisions(r_airbnb=0.01, gamma=0.99), 1.612, places=2)






    def test_c_is_it_better_to_gamble(self):
        from irlc.exam.exam2024spring.question_bill_mdp import c_is_it_better_to_gamble
        self.assertFalse( c_is_it_better_to_gamble(r_airbnb=0.02, gamma=0.99) )
        self.assertTrue( c_is_it_better_to_gamble(r_airbnb=0.01, gamma=0.99) )





class ControlQuestion(UTestCase):
    def test_a_xdot(self):
        x = 2
        a = 1
        from irlc.exam.exam2024spring.question_control import a_xdot
        self.assertAlmostEqual(a_xdot(x, a), -1, places=3)







    def test_b_rk4_simulate(self):
        from irlc.exam.exam2024spring.question_control import b_rk4_simulate
        u = 2
        tF = 3
        self.assertAlmostEqual(b_rk4_simulate(u, tF), -2.09, places=2)







class Exam2024Spring(Report):
    title = "02465 Exam Spring 2024"
    import irlc
    pack_imports = [irlc]
    abbreviate_questions = True

    q1_questions = [
        (InventoryBrideQuestion, 12),
        (MDPQuestion, 12),
        (ControlQuestion, 12),
                     ]

    questions = []
    questions += q1_questions


if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Exam2024Spring())
