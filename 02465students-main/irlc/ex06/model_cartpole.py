# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex04.discrete_control_cost import DiscreteQRCost
import sympy as sym
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from irlc.ex03.control_model import ControlModel
from irlc.ex03.control_cost import SymbolicQRCost
from irlc.ex04.discrete_control_model import DiscreteControlModel
from irlc.ex04.control_environment import ControlEnvironment

class CartpoleModel(ControlModel):
    state_labels = ["$x$", r"$\frac{dx}{dt}$", r"$\theta$", r"$\frac{d \theta}{dt}$"]
    action_labels = ["Cart force $u$"]

    def __init__(self, mc=2,
                     mp=0.5,
                     l=0.5,
                     max_force=50, dist=1.0):
        self.mc = mc
        self.mp = mp
        self.l = l
        self.max_force = max_force
        self.dist = dist
        self.cp_render = {}
        super().__init__()


    def tF_bound(self) -> Box:
        return Box(0.01, np.inf, shape=(1,), dtype=np.float64)

    def x_bound(self) -> Box:
        return Box(np.asarray([-2 * self.dist, -np.inf, -2 * np.pi, -np.inf]), np.asarray([2 * self.dist, np.inf, 2 * np.pi, np.inf]), dtype=np.float64)

    def x0_bound(self) -> Box:
        return Box(np.asarray([0, 0, np.pi, 0]), np.asarray([0, 0, np.pi, 0]), dtype=np.float64)

    def xF_bound(self) -> Box:
        return Box(np.asarray([self.dist, 0, 0, 0]), np.asarray([self.dist, 0, 0, 0]), dtype=np.float64)

    def u_bound(self) -> Box:
        return Box(np.asarray([-self.max_force]), np.asarray([self.max_force]), dtype=np.float64)

    def get_cost(self) -> SymbolicQRCost:
        return SymbolicQRCost(R=np.eye(1) * 0, Q=np.eye(4) * 0, qc=1)  # just minimum time

    def sym_f(self, x, u, t=None):
        mp = self.mp
        l = self.l
        mc = self.mc
        g = 9.81 # Gravity on earth.

        x_dot = x[1]
        theta = x[2]
        sin_theta = sym.sin(theta)
        cos_theta = sym.cos(theta)
        theta_dot = x[3]
        F = u[0]
        # Define dynamics model as per Razvan V. Florian's
        # "Correct equations for the dynamics of the cart-pole system".
        # Friction is neglected.

        # Eq. (23)
        temp = (F + mp * l * theta_dot ** 2 * sin_theta) / (mc + mp)
        numerator = g * sin_theta - cos_theta * temp
        denominator = l * (4.0 / 3.0 - mp * cos_theta ** 2 / (mc + mp))
        theta_dot_dot = numerator / denominator

        # Eq. (24)
        x_dot_dot = temp - mp * l * theta_dot_dot * cos_theta / (mc + mp)
        xp = [x_dot,
              x_dot_dot,
              theta_dot,
              theta_dot_dot]
        return xp

    def close(self):
        for r in self.cp_render.values():
            r.close()

    def render(self, x, render_mode="human"):
        if render_mode not in self.cp_render:
            self.cp_render[render_mode] = gym.make("CartPole-v1", render_mode=render_mode)  # environment only used for rendering. Change to v1 in gym 0.26.
            self.cp_render[render_mode].max_time_limit = 10000
            self.cp_render[render_mode].reset()
        self.cp_render[render_mode].unwrapped.state = np.asarray(x)  # environment is wrapped
        return self.cp_render[render_mode].render()

class SinCosCartpoleModel(CartpoleModel):
    def phi_x(self, x):  
        x, dx, theta, theta_dot = x[0], x[1], x[2], x[3]
        return [x, dx, sym.sin(theta), sym.cos(theta), theta_dot]

    def phi_x_inv(self, x):
        x, dx, sin_theta, cos_theta, theta_dot = x[0], x[1], x[2], x[3], x[4]
        theta = sym.atan2(sin_theta, cos_theta)  # Obtain angle theta from sin(theta),cos(theta)
        return [x, dx, theta, theta_dot]

    def phi_u(self, u):
        return [sym.atanh(u[0] / self.max_force)]

    def phi_u_inv(self, u):
        return [sym.tanh(u[0]) * self.max_force] 

def _cartpole_discrete_cost(model):
    pole_length = model.continuous_model.l

    state_size = model.state_size
    Q = np.eye(state_size)
    Q[0, 0] = 1.0
    Q[1, 1] = Q[4, 4] = 0.
    Q[0, 2] = Q[2, 0] = pole_length
    Q[2, 2] = Q[3, 3] = pole_length ** 2

    print("Warning: I altered the cost-matrix to prevent underflow. This is not great.")
    R = np.array([[0.1]])
    Q_terminal = 1 * Q

    q = np.asarray([0,0,0,-1,0])
    # Instantaneous control cost.
    c3 = DiscreteQRCost(Q=Q*0, R=R * 0.1, q=1 * q, qN=q * 1)
    c3 += c3.goal_seeking_cost(Q=Q, x_target=model.x_upright)
    c3 += c3.goal_seeking_terminal_cost(QN=Q_terminal, xN_target=model.x_upright)
    cost = c3
    return cost

class GymSinCosCartpoleModel(DiscreteControlModel): 
    state_labels =  ['x', 'd_x', r'$\sin(\theta)$', r'$\cos(\theta)$', r'$d\theta/dt$']
    action_labels = ['Torque $u$']

    def __init__(self, dt=0.02, cost=None, transform_actions=True, **kwargs): 
        model = SinCosCartpoleModel(**kwargs)
        self.transform_actions = transform_actions
        super().__init__(model=model, dt=dt, cost=cost) 
        self.x_upright = np.asarray(self.phi_x(model.xF_bound().low ))
        if cost is None:
            cost = _cartpole_discrete_cost(self)
        self.cost = cost

    @property
    def max_force(self):
        return self.continuous_model.maxForce


class GymSinCosCartpoleEnvironment(ControlEnvironment): 
    def __init__(self, Tmax=5, transform_actions=True, supersample_trajectory=False, render_mode='human', **kwargs):
        discrete_model = GymSinCosCartpoleModel(transform_actions=transform_actions, **kwargs)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64)
        if transform_actions:
            self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        super().__init__(discrete_model, Tmax=Tmax,render_mode=render_mode, supersample_trajectory=supersample_trajectory) 


class DiscreteCartpoleModel(DiscreteControlModel):
    def __init__(self, dt=0.02, cost=None, **kwargs):
        model = CartpoleModel(**kwargs)
        super().__init__(model=model, dt=dt, cost=cost)


class CartpoleEnvironment(ControlEnvironment):
    def __init__(self, Tmax=5, supersample_trajectory=False, render_mode='human', **kwargs):
        discrete_model = DiscreteCartpoleModel(**kwargs)
        super().__init__(discrete_model, Tmax=Tmax, supersample_trajectory=supersample_trajectory, render_mode=render_mode)


if __name__ == "__main__":
    from irlc import Agent, train
    env = GymSinCosCartpoleEnvironment(render_mode='human')
    agent = Agent(env)
    stats, traj = train(env, agent, num_episodes=1, max_steps=100)
    env.close()
