# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import matplotlib.pyplot as plt
import numpy as np
from irlc import savepdf, train
from irlc.ex04.pid_locomotive_agent import PIDLocomotiveAgent
from irlc.ex05.lqr_agent import LQRAgent
from irlc.ex04.model_harmonic import HarmonicOscilatorEnvironment
from irlc.ex05.boeing_lqr import compute_A_B_d, compute_Q_R_q

class ConstantLQRAgent(LQRAgent):
    # TODO: 3 lines missing.
    raise NotImplementedError("Complete this agent here. You need to update the policy-function: def pi(self, ..).")

def get_Kp_Kd(L0):
    # TODO: 1 lines missing.
    raise NotImplementedError("Use lqr_agent.L to define Kp and Kd.")
    return Kp, Kd


if __name__ == "__main__":
    Delta = 0.06  # Time discretization constant
    # Define a harmonic osscilator environment. Use .., render_mode='human' to see a visualization.
    env = HarmonicOscilatorEnvironment(Tmax=8, dt=Delta, m=0.5, R=np.eye(1) * 8, render_mode=None)  # set render_mode='human' to see the oscillator.
    model = env.discrete_model.continuous_model # Get the ControlModel corresponding to this environment.


    # Compute the discretized A, B and d matrices using the helper functions we defined in the Boeing problem.
    # Note that these are for the discrete environment: x_{k+1} = A x_k + B u_k + d
    A, B, d = compute_A_B_d(model, Delta)
    Q, R, q = compute_Q_R_q(model, Delta)

    # Run the LQR agent
    lqr_agent = LQRAgent(env, A=A, B=B, d=d, Q=Q, R=R, q=q)
    _, traj1 = train(env, lqr_agent, return_trajectory=True)

    # Part 1. Build an agent that always takes actions u_k = L_0 x_k + l_0
    constant_agent = ConstantLQRAgent(env, A=A, B=B, d=d, Q=Q, R=R, q=q)
    # Check that its policy is independent of $k$:
    x0, _ = env.reset() 
    print(f"Initial state is {x0=}")
    print(f"Action at time step k=0 {constant_agent.pi(x0, k=0)=}")
    print(f"Action at time step k=5 (should be the same) {constant_agent.pi(x0, k=0)=}") 

    _, traj2 = train(env, constant_agent, return_trajectory=True)

    # Part 2. Use the L and l matrices (see lqr_agent.L and lqr_agent.l)
    # to select Kp and Kd in a PID agent. Then let's use the Locomotive agent to see the effect of the controller.
    # Use render_mode='human' to see its effect.
    # We only need to use L.
    # Hint: compare the form of the LQR and PID controller and use that to select Kp and Kd.
    Kp, Kd = get_Kp_Kd(lqr_agent.L[0]) # Use lqr_agent.L to define Kp and Kd.

    # Define and run the PID agent.
    pid_agent = PIDLocomotiveAgent(env, env.dt, Kp=Kp, Kd=Kd)
    _, traj3 = train(env, pid_agent, return_trajectory=True)

    # Plot all actions and state sequences.
    plt.figure(figsize=(10,5))
    plt.grid()
    plt.plot(traj1[0].time[:-1], traj1[0].action, label="Optimal LQR action sequence")
    plt.plot(traj2[0].time[:-1], traj2[0].action, '.-', label="Constant LQR action sequence")
    plt.plot(traj3[0].time[:-1], traj3[0].action, label="PID agent action sequence")
    plt.xlabel("Time / Seconds")
    plt.ylabel("Action / Newtons")
    plt.ylim([-.2, .2])
    plt.legend()
    savepdf("pid_lqr_actions")
    plt.show(block=True)

    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.plot(traj1[0].time, traj1[0].state[:, 0], label="Optimal LQR states x(t)")
    plt.plot(traj2[0].time, traj2[0].state[:, 0], label="Constant LQR states x(t)")
    plt.plot(traj3[0].time, traj3[0].state[:, 0], label="PID agent states x(t)")
    plt.xlabel("Time / Seconds")
    plt.ylabel("Position x(t) / Meters")
    plt.ylim([-1, 1])
    plt.legend()
    savepdf("pid_lqr_states")
