# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""

References:
  [Her25] Tue Herlau. Sequential decision making. (Freely available online), 2025.
"""
import numpy as np
import matplotlib.pyplot as plt
from irlc import savepdf
from irlc import train
from irlc.ex05.model_boeing import BoeingEnvironment
from irlc.ex05.lqr_agent import LQRAgent
from irlc.ex03.control_model import ControlModel
import scipy


def boeing_simulation():
    env = BoeingEnvironment(Tmax=10)
    model = env.discrete_model.continuous_model # get the model from the Boeing environment
    dt = env.dt # Get the discretization time.
    A, B, d = compute_A_B_d(model, dt)
    # Use compute_Q_R_q to get the Q, R, and q matrices in the discretized system
    # TODO: 1 lines missing.
    raise NotImplementedError("Compute Q, R and q here")
    ## TODO: Half of each line of code in the following 1 lines have been replaced by garbage. Make it work and remove the error.
    #----------------------------------------------------------------------------------------------------------------------------
    # agent = LQRAgent(env, A=A??????????????????????????
    raise NotImplementedError("Use your LQRAgent to plan using the system matrices.")
    stats, trajectories = train(env, agent, return_trajectory=True)
    return stats, trajectories, env

def compute_Q_R_q(model : ControlModel, dt : float):
    cost = model.get_cost() # Get the continuous-time cost-function
    # use print(cost) to see what it contains.
    # Then get the discretized matrices using the techniques described in (Her25, Subsection 13.1.6).
    # TODO: 3 lines missing.
    raise NotImplementedError("Insert your solution and remove this error.")
    return Q, R, q

def compute_A_B_d(model : ControlModel, dt : float):
    if model.d is None:
        d = np.zeros((model.state_size,))  # Ensure d is set to a zero vector if it is not defined.
    else:
        d = model.d

    A_discrete = scipy.linalg.expm(model.A * dt)  # This is the discrete A-matrix computed using the matrix exponential
    # Now it is your job to define B_discrete and d_discrete.
    # TODO: 2 lines missing.
    raise NotImplementedError("Insert your solution and remove this error.")
    return A_discrete, B_discrete, d_discrete.flatten()

def boeing_experiment():
    _, trajectories, env = boeing_simulation()
    model = env.discrete_model.continuous_model

    dt = env.dt 
    Q, R, q = compute_Q_R_q(model, dt)
    print("Discretization time is", dt)
    print("Original q-vector was:", model.get_cost().q)
    print("Discretized version is:", q) 

    t = trajectories[-1]
    out = t.state @ model.P.T

    plt.plot(t.time, out[:, 0], '-', label=env.observation_labels[0])
    plt.plot(t.time, out[:, 1], '-', label=env.observation_labels[1])
    plt.grid()
    plt.legend()
    plt.xlabel("Time/seconds")
    plt.ylabel("Output")
    savepdf("boing_lqr_output")
    plt.show(block=False)
    plt.close()

    plt.plot(t.time[:-1], t.action[:, 0], '-', label=env.action_labels[0])
    plt.plot(t.time[:-1], t.action[:, 1], '-', label=env.action_labels[1])
    plt.xlabel("Time/seconds")
    plt.ylabel("Control action")
    plt.grid()
    plt.legend()
    savepdf("boing_lqr_action")
    plt.show()

if __name__ == "__main__":
    boeing_experiment()
