from irlc.ex04.model_pendulum import PendulumModel
from irlc.ex04.discrete_control_model import DiscreteControlModel
from irlc.exam.exam2023spring.dlqr import LQR
import numpy as np

# TODO: Code has been removed from here.
raise NotImplementedError("Insert your solution and remove this error.")

def a_LQR_solve(a : float, x0 : np.ndarray) -> float:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return u

def b_linearize(theta : float):
    # TODO: 7 lines missing.
    raise NotImplementedError("Insert your solution and remove this error.")
    return A, B, d


def c_get_optimal_linear_policy(x0 : np.ndarray) -> float:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return u

if __name__ == "__main__":
    theta = np.pi/2  # An example: linearize around theta = pi/2.
    a = 1
    x0 = np.asarray([1, 0])
    print(f"a) LQR action should be approximately -1.666, you got: {a_LQR_solve(a, x0)=}")
    A, B, d = b_linearize(theta) # Get the three matrices.
    print(f"b) Entry d[1] should be approx. 4.91, you got: {d[1]=}")
    theta = 0.1  # Try a small initial angle.
    print(f"c) Optimal policy for linearized problem should be approximately -1.07, you got: {c_get_optimal_linear_policy(x0=np.asarray([theta, 0]))=}")
