def pid(xs : list, xstar :float , Kp=0., Ki=0., Kd=0., stable=False): 
    us = []
    e_prev = 0
    es = []
    I = 0
    Delta = 1
    for k, x in enumerate(xs):
        e =  xstar - x
        es.append(e)

        I = I + Delta * e

        if k > 2 and stable:
            d1 = (es[-1] - es[-2])/Delta
            d2 = (es[-2] - es[-3]) / Delta

            dterm = (d1+d2)/2
        else:
            dterm = (e-e_prev)/ Delta

        u = Kp * e + Ki * I + Kd * dterm
        e_prev = e
        us.append(u)
    return us[-1] 

def a_pid_Kp(xs : list, xstar : float, Kp : float) -> float:
    u = pid(xs, xstar, Kp=Kp) 
    return u

def b_pid_full(xs : list, xstar : float, Kp : float, Ki : float, Kd : float) -> float:
    u = pid(xs, xstar, Kp=Kp, Ki=Ki, Kd=Kd) 
    return u

def c_pid_stable(xs : list, xstar : float, Kp : float, Ki : float, Kd : float) -> float:
    u = pid(xs, xstar, Kp=Kp, Ki=Ki, Kd=Kd, stable=True) 
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