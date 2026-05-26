
def a_pid_Kp(xs : list, xstar : float, Kp : float) -> float:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return u

def b_pid_full(xs : list, xstar : float, Kp : float, Ki : float, Kd : float) -> float:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return u

def c_pid_stable(xs : list, xstar : float, Kp : float, Ki : float, Kd : float) -> float:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return u


if __name__ == "__main__":
    xs = [10, 8, 7, 5, 3, 1, 0, -2, -1, 0, 2] # Sequence of inputs x_k
    Kp = 0.5
    Ki = 0.05
    Kd = 0.25
    xstar = -1
    u_a = a_pid_Kp(xs, xstar=0, Kp=Kp)
    print(f"Testing part a. Got {u_a}, expected -1.")

    u_b = b_pid_full(xs, xstar=-1, Kp=Kp, Ki=Ki, Kd=Kd)
    print(f"Testing part b. Got {u_b}, expected -4.2")

    u_c = c_pid_stable(xs, xstar=-1, Kp=Kp, Ki=Ki, Kd=Kd)
    print(f"Testing part c. Got {u_c}, expected -4.075")
