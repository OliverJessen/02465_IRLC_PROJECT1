# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from irlc.utils.graphics_util_pygame import UpgradedGraphicsUtil
from irlc.ex04.discrete_control_model import DiscreteControlModel
from irlc.ex04.control_environment import ControlEnvironment
from irlc.ex04.model_linear_quadratic import LinearQuadraticModel
from gymnasium.spaces import Box

"""
SEE: https://github.com/anassinator/ilqr/blob/master/examples/rendezvous.ipynb
"""
class ContiniousRendevouzModel(LinearQuadraticModel): 
    state_labels= ["x0", "y0", "x1", "y1", 'Vx0', "Vy0", "Vx1", "Vy1"]
    action_labels = ['Fx0', 'Fy0', "Fx1", "Fy1"]
    x0 = np.array([0, 0, 10, 10, 0, -5, 5, 0])  # Initial state.

    def __init__(self, m=10.0, alpha=0.1, simple_bounds=None, cost=None): 
        m00 = np.zeros((4,4))
        mI = np.eye(4)
        A = np.block( [ [m00, mI], [m00, -alpha/m*mI] ] )
        B = np.block( [ [m00], [mI/m]] )
        state_size = len(self.x0)
        action_size = 4
        self.m = m
        self.alpha = alpha
        Q = np.eye(state_size)
        Q[0, 2] = Q[2, 0] = -1
        Q[1, 3] = Q[3, 1] = -1
        R = 0.1 * np.eye(action_size)
        self.viewer = None
        super().__init__(A=A, B=B, Q=Q*20, R=R*20)

    def x0_bound(self) -> Box:
        return Box(self.x0, self.x0) # self.bounds['x0_low'] = self.bounds['x0_high'] = list(self.x0)

    def render(self, x, render_mode="human"):
        """ Render the environment. You don't have to understand this code.  """
        if self.viewer is None:
            self.viewer = HarmonicViewer(xstar=0, x0=self.x0) # target: x=0.
        self.viewer.update(x)
        import time
        time.sleep(0.05)
        return self.viewer.blit(render_mode=render_mode)

    def close(self):
        pass


class DiscreteRendevouzModel(DiscreteControlModel): 
    def __init__(self, dt=0.1, cost=None, transform_actions=True, **kwargs):
        model = ContiniousRendevouzModel(**kwargs)
        super().__init__(model=model, dt=dt, cost=cost) 

class RendevouzEnvironment(ControlEnvironment): 
    def __init__(self, Tmax=20, render_mode=None, **kwargs):
        discrete_model = DiscreteRendevouzModel(**kwargs)
        super().__init__(discrete_model, Tmax=Tmax, render_mode=render_mode) 

class HarmonicViewer(UpgradedGraphicsUtil):
    def __init__(self, xstar = 0, x0=None):
        self.xstar = xstar
        width = 800
        self.x0 = x0
        sz = 20
        self.scale = width/(2*sz)
        self.p1h = []
        self.p2h = []
        super().__init__(screen_width=width, xmin=-sz, xmax=sz, ymin=-sz, ymax=sz, title='Rendevouz environment')

    def render(self):
        self.draw_background(background_color=(255, 255, 255))
        # dw = self.dw
        p1 = self.x[:2]
        p2 = self.x[2:4]
        self.p1h.append(p1)
        self.p2h.append(p2)
        self.circle("asdf", pos=p1, r=.5 * self.scale, fillColor=(200, 0, 0))
        self.circle("asdf", pos=p2, r=.5 * self.scale, fillColor=(0, 0, 200) )
        if len(self.p1h) > 2:
            self.polyline('...', np.stack(self.p1h)[:,0], np.stack(self.p1h)[:,1], width=1, color=(200, 0, 0))
            self.polyline('...', np.stack(self.p2h)[:,0], np.stack(self.p2h)[:,1], width=1, color=(0, 0, 200))

        if tuple(self.x) == tuple(self.x0):
            self.p1h = []
            self.p2h = []


    def update(self, x):
        self.x = x


if __name__ == "__main__":
    from irlc import Agent, train
    env = RendevouzEnvironment(render_mode='human')
    from irlc.ex05.lqr_agent import LQRAgent
    a2 = LQRAgent(env=env)

    stats, traj = train(env, Agent(env), num_episodes=1)
    pass
