# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import sympy as sym
from irlc.ex03.control_model import ControlModel
from irlc.ex03.control_cost import SymbolicQRCost
import numpy as np

class Toy2DControl(ControlModel):
    def get_cost(self):
        # You get the cost-function for free because it can be anything as far as this problem is concerned.
        return SymbolicQRCost(Q=np.eye(2), R=np.eye(1))

    def sym_f(self, x, u, t=None): 
        return [x[1], sym.cos(x[0] + u[0])] 

def toy_simulation(u0 : float, T : float) -> float:
    toy = Toy2DControl() 
    x0 = np.asarray([np.pi/2, 0])
    xs, us, ts, cost = toy.simulate( x0=x0, u_fun = u0, t0=0, tF=T)
    wT = xs[-1][0] 
    return wT

if __name__ == "__main__":
    x0 = np.asarray([np.pi/2, 0])
    wT = toy_simulation(u0=0.4, T=5)
    print(f"Starting in x0=[pi/2, 0], after T=5 seconds the system is an an angle {wT=} (should be 1.265)") 
