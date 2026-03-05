# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex04.locomotive import LocomotiveEnvironment
from irlc.ex04.pid_locomotive_agent import PIDLocomotiveAgent
from irlc.ex01.agent import train

def pidplot(Kp=40, Kd=0, Ki=0, slope=0, target=0):
    dt = .04
    m = 70
    Tmax=20
    env = LocomotiveEnvironment(m=m, slope=slope, dt=dt, Tmax=Tmax, render_mode='human')
    agent = PIDLocomotiveAgent(env, dt=dt, Kp=Kp, Ki=Ki, Kd=Kd, target=0)
    train(env, agent, num_episodes=1)
    env.close()

if __name__ == "__main__":
    pidplot(Kp=5, Kd=0, Ki=0)
