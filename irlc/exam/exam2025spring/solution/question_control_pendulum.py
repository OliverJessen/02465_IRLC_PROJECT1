import numpy as np

def a_dynamics_f(theta, thetadot, u) -> list[float, float]:
    f1f2 = [thetadot, float(1.25 * u + 9.82 * np.sin(theta)) ] 
    return f1f2

def b_euler(theta0 : float, thetadot0 : float, delta : float, N : int) -> float:
    x = np.asarray([theta0, thetadot0]) 
    for _ in range(N):
        x = x + delta * np.asarray(a_dynamics_f(x[0], x[1], 0) )
    theta_N = float(x[0]) 
    return theta_N

if __name__ == "__main__":
    print(f"a) f(x, u) should be [0, 11.07], you got: {a_dynamics_f(theta=np.pi/2, thetadot=0, u=1)=}")
    print(f"b) The value of theta after N=3 euler steps should be approx 0.8353, you got {b_euler(theta0=0.1, thetadot0=0, delta=0.5, N=3)=}")