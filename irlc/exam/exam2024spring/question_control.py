import numpy as np
import sympy as sym
from irlc.ex03.control_model import ControlModel
from irlc.ex03.control_cost import SymbolicQRCost

# TODO: Code has been removed from here.
raise NotImplementedError("Insert your solution and remove this error.")

def a_xdot(x : float, a : float) -> float:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return xdot

def b_rk4_simulate(u0 : float, tF : float):
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return xF

if __name__ == "__main__":
    print(f"a): dx/dt should be -1, you got {a_xdot(x=2, a=1)=}")
    print(f"b): Final position x(tF) should be approximately -2.09, you got {b_rk4_simulate(u0=2, tF=3)=}")
