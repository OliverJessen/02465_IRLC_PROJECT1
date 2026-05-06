import numpy as np
import sympy as sym
from irlc.ex03.control_model import ControlModel
from irlc.ex03.control_cost import SymbolicQRCost

class Simulation(ControlModel): 
    def sym_f(self, x, u, t=None):
        return [-sym.exp( u[0] -x[0]**2 )]

    def get_cost(self): # The cost is only required to specify dimensions of x and u.
        return SymbolicQRCost(Q=np.eye(1), R=np.eye(1)) 

def a_xdot(x : float, a : float) -> float:
    m = Simulation() 
    u = a * x**2 # This approach validates our implementation of the system. A manual implementation is just as good.
    xd_ = -np.exp( u - x**2 )
    xdot = m.f((x,), (u,), 0)[0]
    assert xd_ == xdot 
    return xdot

def b_rk4_simulate(u0 : float, tF : float):
    x = 0  
    m = Simulation()
    xs, us, ts, J_ = m.simulate((x,), u_fun=(u0,), t0=0, tF=tF)
    xF =xs[-1][0] 
    return xF

if __name__ == "__main__":
    print(f"a): dx/dt should be -1, you got {a_xdot(x=2, a=1)=}")
    print(f"b): Final position x(tF) should be approximately -2.09, you got {b_rk4_simulate(u0=2, tF=3)=}")