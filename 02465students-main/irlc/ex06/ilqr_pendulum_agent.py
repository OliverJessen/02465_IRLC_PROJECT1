# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from irlc.ex04.model_pendulum import GymSinCosPendulumEnvironment
from irlc.ex06.ilqr_agent import ILQRAgent
from irlc import train
from irlc import savepdf
import matplotlib.pyplot as plt

Tmax = 3
def pen_experiment(N=12, use_linesearch=True,figex="", animate=True):
    dt = Tmax / N
    env = GymSinCosPendulumEnvironment(dt, Tmax=Tmax, supersample_trajectory=True, render_mode="human" if animate else None)
    agent = ILQRAgent(env, env.discrete_model, N=N, ilqr_iterations=200, use_linesearch=use_linesearch)

    stats, trajectories = train(env, agent, num_episodes=1, return_trajectory=True)

    agent.use_ubar = True
    stats2, trajectories2 = train(env, agent, num_episodes=1, return_trajectory=True)
    env.close()

    plot_pendulum_trajectory(env, trajectories[0], label='Closed-loop $\\pi$')
    xb = agent.xbar
    tb = np.arange(N+1)*dt
    plt.figure(figsize=(12, 6))
    plt.plot(trajectories[0].time, trajectories[0].state[:,1], '-', label='Closed-loop $\\pi(x_k)$')

    plt.plot(trajectories2[0].time, trajectories2[0].state[:,1], '-', label='Open-loop $\\bar{u}_k$')
    plt.plot(tb, xb[:,1], 'o-', label="iLQR prediction $\\bar{x}_k$")
    plt.grid()
    plt.legend()
    ev = "pendulum"
    savepdf(f"irlc_pendulum_theta_N{N}_{use_linesearch}{figex}")
    plt.show()

    ## Plot J
    plt.figure(figsize=(6, 6))
    plt.semilogy(agent.J_hist, 'k.-')
    plt.xlabel("iLQR Iterations")
    plt.ylabel("Cost function estimate $J$")
    # plt.title("Last value: {")
    plt.grid()
    savepdf(f"irlc_pendulum_J_N{N}_{use_linesearch}{figex}")
    plt.show()

def plot_pendulum_trajectory(env, traj, style='k.-', label=None, action=False, **kwargs):
    if action:
        y = traj.action[:, 0]
        y = np.clip(y, env.action_space.low[0], env.action_space.high[0])
    else:
        y = traj.state[:, 1]

    plt.plot(traj.time[:-1] if action else traj.time, y, style, label=label, **kwargs)
    plt.xlabel("Time/seconds")
    if action:
        plt.ylabel("Torque $u$")
    else:
        plt.ylabel(r"$\cos(\theta)$")
    plt.grid()
    pass

N = 50
if __name__ == "__main__":
    pen_experiment(N=N, use_linesearch=True)
