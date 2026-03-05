# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import matplotlib.pyplot as plt
import numpy as np
from irlc import savepdf, train
from irlc.ex04.pid_locomotive_agent import PIDLocomotiveAgent
from irlc.ex05.lqr_agent import LQRAgent
from irlc.ex04.model_harmonic import HarmonicOscilatorEnvironment
from irlc.ex05.boeing_lqr import compute_A_B_d, compute_Q_R_q
from irlc.ex06.linearization_agent import LinearizationAgent
from irlc.ex05.lqr_pid import ConstantLQRAgent
from irlc.ex04.locomotive import LocomotiveEnvironment
from irlc.ex04.pid_locomotive_agent import PIDLocomotiveAgent
from irlc.ex01.agent import train
from irlc.ex03.control_cost import SymbolicQRCost
import matplotlib
#matplotlib.use('qtagg')
dt = .04
m = 70
Tmax=10
slope = 0

env = LocomotiveEnvironment(m=m, slope=slope, dt=dt, Tmax=Tmax, render_mode='human')

model = env.discrete_model
model.cost = SymbolicQRCost(Q=np.eye(2)*10000, R=np.eye(1)).discretize(dt=dt)
agent = LinearizationAgent(env, model=model, xbar=env.observation_space.sample(), ubar=env.action_space.sample())
_, traj = train(env, agent, num_episodes=1)
env.close()
if False:
    from irlc import plot_trajectory, savepdf
    import matplotlib.pyplot as plt
    plt.figure()
    plot_trajectory(trajectory=traj[0], env=env, xkeys=[0, 1], ukeys=[])
    savepdf('lqr_pid_locomotive_state.pdf')
    plot_trajectory(trajectory=traj[0], env=env, ukeys=[0], xkeys=[])
    savepdf('lqr_pid_locomotive_action.pdf')
    env.close()
