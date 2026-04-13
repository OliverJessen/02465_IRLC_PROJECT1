# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import UTestCase, Report
import irlc

class JarJarPiOptimal(UTestCase):
    """ Problem 1: Compute optimal policy.  """
    def test_pi_1(self):
        from irlc.project3.jarjar import pi_optimal
        self.assertLinf(pi_optimal(1), -1)

    def test_pi_all(self):
        from irlc.project3.jarjar import pi_optimal
        for s in range(-10, 10):
            if s != 0:
                self.assertLinf(pi_optimal(s))

class JarJarQ0Estimated(UTestCase):
    """ Problem 2: Implement Q0_approximate to (approximate) the Q-function for the optimal policy.  """
    def test_Q0_N1(self):
        from irlc.project3.jarjar import Q0_approximate
        import numpy as np
        self.assertLinf(np.abs(Q0_approximate(gamma=0.8, N=1))) # TODO: Remove abs. This was added due to typo.

    def test_Q0_N2(self):
        from irlc.project3.jarjar import Q0_approximate
        import numpy as np
        self.assertLinf(np.abs(Q0_approximate(gamma=0.7, N=20))) # TODO: Remove abs. This was added due to typo.

    def test_Q0_N100(self):
        from irlc.project3.jarjar import Q0_approximate
        import numpy as np
        self.assertLinf(np.abs(Q0_approximate(gamma=0.9, N=20)))  # TODO: Remove abs. This was added due to typo.


class JarJarQExact(UTestCase):
    """ Problem 4: Compute Q^*(s,a) exactly by extending analytical solution. """
    def test_Q_s0(self):
        from irlc.project3.jarjar import Q_exact
        self.assertLinf(Q_exact(0, gamma=0.8, a=1))
        self.assertLinf(Q_exact(0, gamma=0.8, a=-1))

    def test_Q_s1(self):
        from irlc.project3.jarjar import Q_exact
        self.assertLinf(Q_exact(1, gamma=0.8, a=-1))
        self.assertLinf(Q_exact(1, gamma=0.95, a=-1))
        self.assertLinf(Q_exact(1, gamma=0.7, a=-1))

    def test_Q_s_positive(self):
        from irlc.project3.jarjar import Q_exact
        for s in range(20):
            self.assertLinf(Q_exact(s, gamma=0.75, a=-1))

    def test_Q_all(self):
        from irlc.project3.jarjar import Q_exact
        for s in range(-20, 20):
            self.assertLinf(Q_exact(s, gamma=0.75, a=-1))
            self.assertLinf(Q_exact(s, gamma=0.75, a=1))

class Project3(Report):
    title = "Project part 3: Reinforcement Learning"
    pack_imports = [irlc]

    jarjar1 = [(JarJarPiOptimal, 10),
               (JarJarQ0Estimated, 10),
               (JarJarQExact, 10) ]
    questions = []
    questions += jarjar1
    # questions += rebels

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Project3())
