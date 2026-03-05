# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex04.locomotive import LocomotiveEnvironment
from irlc import train, plot_trajectory, savepdf, Agent
from irlc.ex05.dlqr import LQR
from irlc.ex04.control_environment import ControlEnvironment
import numpy as np
import matplotlib.pyplot as plt

class LQRAgent(Agent):
    def __init__(self, env : ControlEnvironment, A, B, Q, R, d=None, q=None):
        N = int((env.Tmax / env.dt)) # Obtain the planning horizon
        """ Define A, B as the list of A/B matrices here. I.e. x[t+1] = A x[t] + B x[t] + d.
        You should use the function model.f to do this, which has build-in functionality to compute Jacobians which will be equal to A, B """
        """ Define self.L, self.l here as the (lists of) control matrices. """
        ## TODO: Half of each line of code in the following 1 lines have been replaced by garbage. Make it work and remove the error.
        #----------------------------------------------------------------------------------------------------------------------------
        # (self.L, self.l), _ = LQR(A=[A]*N, B=[B]*N, d=[d]*N if d is not No???????????????????????????????????????????????????????????????????
        raise NotImplementedError("Insert your solution and remove this error.")
        self.dt = env.dt
        super().__init__(env)

    def pi(self,x, k, info=None):
        """
        Compute the action here using u = L_k x + l_k.
        You should use self.L, self.l to get the control matrices (i.e. L_k = self.L[k] ),
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Compute current action here")
        return u


if __name__ == "__main__":
    # Make a guess at the system matrices for planning. We will return on how to compute these exactly in a later exercise.
    A = np.ones((2, 2))
    A[1, 0] = 0
    B = np.asarray([[0], [1]])
    Q = np.eye(2)*3
    R = np.ones((1, 1))*2
    q = np.asarray([-1.1, 0 ])

    # Create and test our LQRAgent.
    env = LocomotiveEnvironment(render_mode='human', Tmax=10, slope=1)
    agent = LQRAgent(env, A=A, B=B, Q=Q, R=R, q=q)
    stats, traj = train(env, agent, num_episodes=1)

    env.reset()
    savepdf("locomotive_snapshot.pdf", env=env) # Make a plot for the exercise file.
    env.state_labels = ["x(t)", "v(t)"]
    env.action_labels = ["u(t)"]
    plot_trajectory(traj[0], env)
    plt.show(block=True)
    savepdf("lqr_agent")
    plt.show()
    env.close()
