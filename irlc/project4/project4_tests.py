# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import UTestCase, Report
import irlc


class RebelsSimple(UTestCase):
    """ Problem 1: Test the UCB-algorithm in the basic-environment with a single state """
    def test_simple_four_episodes(self):
        """ Test the first four episodes in the simple grid problem. """
        from irlc.project4.rebels import get_ucb_actions, very_basic_grid
        actions = get_ucb_actions(very_basic_grid, alpha=0.1, episodes=4, c=5, plot=False)
        # Make sure we only have 4 actions (remember to truncate the action-sequences!)
        self.assertEqual(len(actions), 4) # Check the number of actions are correct
        self.assertEqual(actions[0], 0) # Check the first action is correct
        self.assertEqualC(actions) # Check all actions.

    def test_simple_nine_episodes(self):
        """ Test the first nine episodes in the simple grid problem. """
        from irlc.project4.rebels import get_ucb_actions, very_basic_grid
        actions = get_ucb_actions(very_basic_grid, alpha=0.1, episodes=9, c=5, plot=False)
        self.assertEqual(len(actions), 9) # Check the number of actions are correct
        self.assertEqual(actions[0], 0) # Check the first action is correct
        self.assertEqualC(actions) # Check all actions.

    def test_simple_environment(self):
        from irlc.project4.rebels import get_ucb_actions, very_basic_grid
        actions = get_ucb_actions(very_basic_grid, alpha=0.1, episodes=100, c=5, plot=False)
        # Check the number of actions are correct
        self.assertEqualC(len(actions))
        # Check the first action is correct
        self.assertEqualC(actions[0])
        # Check all actions.
        self.assertEqualC(actions)

    def test_bridge_environment(self):
        from irlc.gridworld.gridworld_environments import grid_bridge_grid
        from irlc.project4.rebels import get_ucb_actions, very_basic_grid
        actions = get_ucb_actions(grid_bridge_grid, alpha=0.1, episodes=1000, c=2, plot=False)
        self.assertEqualC(len(actions))
        # Check all actions.
        self.assertEqualC(actions)

class RebelsBridge(UTestCase):
    """ Problem 1: Test the UCB-algorithm in the bridge-environment """
    def test_bridge_environment_one(self):
        from irlc.gridworld.gridworld_environments import grid_bridge_grid
        from irlc.project4.rebels import get_ucb_actions
        actions = get_ucb_actions(grid_bridge_grid, alpha=0.1, episodes=1, c=2, plot=False)
        self.assertEqualC(len(actions))
        self.assertEqualC(actions)

    def test_bridge_environment_two(self):
        from irlc.gridworld.gridworld_environments import grid_bridge_grid
        from irlc.project4.rebels import get_ucb_actions
        actions = get_ucb_actions(grid_bridge_grid, alpha=0.1, episodes=2, c=2, plot=False)
        self.assertEqualC(len(actions))
        self.assertEqualC(actions)

    def test_bridge_environment_short(self):
        from irlc.gridworld.gridworld_environments import grid_bridge_grid
        from irlc.project4.rebels import get_ucb_actions
        actions = get_ucb_actions(grid_bridge_grid, alpha=0.1, episodes=30, c=2, plot=False)
        self.assertEqualC(len(actions))
        self.assertEqualC(actions)

    def test_bridge_environment_long(self):
        from irlc.gridworld.gridworld_environments import grid_bridge_grid
        from irlc.project4.rebels import get_ucb_actions
        actions = get_ucb_actions(grid_bridge_grid, alpha=0.1, episodes=1000, c=2, plot=False)
        self.assertEqualC(len(actions))
        self.assertEqualC(actions)



class ProbeLinearApproximations(UTestCase):
    @classmethod
    def setUpClass(cls):
        from irlc.project4.probe_droids import linear_experiment
        super().setUpClass()
        episodes = 20000
        alpha = 0.02
        states_and_actions = [((0, 0), 0),
                              ((0, 0), 1),
                              ((2, 0), 3),
                              ((0, 2), 1),
                              ]

        # Expensive calls: run once only
        cls.linear_qs = linear_experiment(
            episodes=episodes,
            alpha=alpha,
            states_and_actions=states_and_actions,
        )

    def test_linear_q_00_a0(self):
        true_value = 0.315 # Estimate
        q = self.linear_qs[((0, 0), 0)]
        self.assertAlmostEqual(q, true_value, delta=0.25)

    def test_linear_q_02_a1(self):
        true_value = 0.544 # Estimate
        q = self.linear_qs[((0, 2), 1)]
        self.assertAlmostEqual(q, true_value, delta=0.25)



class ProbeQuadraticApproximations(UTestCase):
    @classmethod
    def setUpClass(cls):
        from irlc.project4.probe_droids import quadratic_experiment

        super().setUpClass()

        episodes = 20000
        alpha = 0.02

        states_and_actions = [((0, 0), 0),
                              ((0, 0), 1),
                              ((2, 0), 3),
                              ((0, 2), 1),
                              ]

        cls.quadratic_qs = quadratic_experiment(
            episodes=episodes,
            alpha=alpha,
            states_and_actions=states_and_actions,
        )

    def test_quadratic_q_00_a1(self):
        true_value = -0.250  # Estimate
        q = self.quadratic_qs[((0, 0), 1)]
        self.assertAlmostEqual(q, true_value, delta=0.20)

    def test_quadratic_q_02_a1(self):
        true_value = 0.788  # Estimate
        q = self.quadratic_qs[((0, 2), 1)]
        self.assertAlmostEqual(q, true_value, delta=0.20)


class Project4(Report):
    title = "Project part 4: Reinforcement Learning II"
    pack_imports = [irlc]


    rebels = [(RebelsSimple, 20),
              (RebelsBridge, 20),
              ]

    probe = [ (ProbeLinearApproximations, 20),
              (ProbeQuadraticApproximations, 20),]

    questions = []
    questions += rebels
    questions += probe

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Project4())
