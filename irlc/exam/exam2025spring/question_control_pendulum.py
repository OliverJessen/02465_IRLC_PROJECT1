import numpy as np

def a_dynamics_f(theta, thetadot, u) -> list[float, float]:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return f1f2

def b_euler(theta0 : float, thetadot0 : float, delta : float, N : int) -> float:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return theta_N

if __name__ == "__main__":
    print(f"a) f(x, u) should be [0, 11.07], you got: {a_dynamics_f(theta=np.pi/2, thetadot=0, u=1)=}")
    print(f"b) The value of theta after N=3 euler steps should be approx 0.8353, you got {b_euler(theta0=0.1, thetadot0=0, delta=0.5, N=3)=}")
