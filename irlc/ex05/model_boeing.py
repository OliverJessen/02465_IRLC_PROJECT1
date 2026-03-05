# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from irlc.ex04.discrete_control_model import DiscreteControlModel
from irlc.ex04.control_environment import ControlEnvironment
from irlc.ex04.model_linear_quadratic import LinearQuadraticModel

class BoeingModel(LinearQuadraticModel):
    """
    Boeing 747 level flight example.

    See: https://books.google.dk/books?id=tXZDAAAAQBAJ&pg=PA147&lpg=PA147&dq=boeing+747+flight+0.322+model+longitudinal+flight&source=bl&ots=L2RpjCAWiZ&sig=ACfU3U2m0JsiHmUorwyq5REcOj2nlxZkuA&hl=en&sa=X&ved=2ahUKEwir7L3i6o3qAhWpl4sKHQV6CdcQ6AEwAHoECAoQAQ#v=onepage&q=boeing%20747%20flight%200.322%20model%20longitudinal%20flight&f=false
    Also: https://web.stanford.edu/~boyd/vmls/vmls-slides.pdf
    """
    state_labels = ["Longitudinal velocity (x) ft/sec", "Velocity in y-axis ft/sec", "Angular velocity",
                    "angle wrt. horizontal"]
    action_labels = ['Elevator', "Throttle"]
    observation_labels = ["Airspeed", "Climb rate"]

    def __init__(self, output=None):
        if output is None:
            output = [10, 0]
        # output = [10, 0]
        A = [[-0.003, 0.039, 0, -0.322],
             [-0.065, -.319, 7.74, 0],
             [.02, -.101, -0.429, 0],
             [0, 0, 1, 0]]
        B = [[.01, 1],
             [-.18, -.04],
             [-1.16, .598],
             [0, 0]]

        A, B = np.asarray(A), np.asarray(B)
        self.u0 = 7.74  # speed in hundred feet/seconds
        self.P = np.asarray([[1, 0, 0, 0], [0, -1, 0, 7.74]])  # Projection of state into airspeed

        dt = 0.1 # Scale the cost by this factor.

        # Set up the cost:
        self.Q_obs = np.eye(2)
        Q = self.P.T @ self.Q_obs @ self.P / dt
        R = np.eye(2) / dt
        q = -np.asarray(output) @ self.Q_obs @ self.P / dt
        super().__init__(A=A, B=B, Q=Q, R=R, q=q)

    def state2outputs(self, x):
        return self.P @ x

class DiscreteBoeingModel(DiscreteControlModel):
    def __init__(self, output=None):
        model = BoeingModel(output=output)
        dt = 0.1
        super().__init__(model=model, dt=dt)


class BoeingEnvironment(ControlEnvironment):
    @property
    def observation_labels(self):
        return self.discrete_model.continuous_model.observation_labels

    def __init__(self, Tmax=10):
        model = DiscreteBoeingModel()
        super().__init__(discrete_model=model, Tmax=Tmax)
