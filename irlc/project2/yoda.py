# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from scipy.linalg import expm  # Computes the matrix exponential e^A for a square matrix A
from numpy.linalg import matrix_power  # Computes A^n for matrix A and integer n


def get_A_B(g : float, L: float, m=0.1): 
    r""" Compute the two matrices A, B (see Problem 1) here and return them.
    The matrices should be numpy ndarrays. """
    A = np.array([[0, 1], [-g/L, 0]], dtype=float)
    B = np.array([[0], [1/(m*L**2)]], dtype=float)
    return A, B


def A_euler(g : float,L : float, Delta : float) -> np.ndarray: 
    r""" Compute \tilde{A}_0 (Euler discretization), see Problem 2.

    Hints:
        * get_A_B can perhaps save you a line or two.
    """
    A, _ = get_A_B(g, L)
    A0_tilde = np.eye(2) + Delta * A
    return A0_tilde

def A_ei(g : float,L : float, Delta : float) -> np.ndarray: 
    r""" Compute A_0 (Exponential discretization), see Problem 2

    Hints:
        * The special function expm(X) computes the matrix exponential e^X. See the lecture notes for more information.
    """
    A, _ = get_A_B(g, L)
    A0 = expm(Delta * A)
    return A0

def M_euler(g : float, L : float, Delta : float, N : int) -> np.ndarray: 
    r""" Compute \tilde{M} (Euler discretization), see Problem 3
    Hints:
        * the matrix_power(X,n) function can compute expressions such as X^n where X is a square matrix and n is a number
    """
    M_tilde = matrix_power(A_euler(g, L, Delta), N)
    return M_tilde

def M_ei(g : float,L : float, Delta : float, N : int) -> np.ndarray: 
    r""" Compute M (Exponential discretization), see Problem 3 """
    M = matrix_power(A_ei(g, L, Delta), N)
    return M

def xN_bound_euler(g : float, L : float,Delta : float,N : int) -> float: 
    r""" Compute upper bound on |x_N| when using Euler discretization, see Problem 6.
    The function should just return a number.

    Hints:
        * This function uses all input arguments.
    """
    bound = np.max(np.abs(np.linalg.eigvals(M_euler(g, L, Delta, N))))
    return bound

def xN_bound_ei(g: float,L : float,Delta : float,N : int) -> float: 
    r""" Compute upper bound on |x_N| when using exponential discretization, see Problem 7.

    Hints:
        * This function does NOT use all input arguments.
        * This will be the hardest problem to solve, but the easiest function to implement.
    """
    bound = np.max(np.sqrt(g/L), 1/np.sqrt(g/L))
    return bound

if __name__ == '__main__':
    g = 9.82 # gravitational constant
    L = 5 # Length of string
    m = 0.1 # Mass of pendulum (in kg)
    Delta = 0.3 # Time-discretization constant Delta (in seconds)
    N = 100 # Time steps

    # Solve Problem 2
    print("A0_euler")
    print(A_euler(g, L, Delta))

    print("A0_ei")
    print(A_ei(g, L, Delta))

    # Solve Problem 3
    print("M_euler")
    print(M_euler(g, L, Delta, N))

    print("M_ei")
    print(M_ei(g, L, Delta, N))

    # Solve Problem 7, upper bound on x_N using Euler discretization
    print("|x_N| <= ", xN_bound_euler(g, L, Delta, N))

    # Solve Problem 8, upper bound on x_N using Exponential discretization
    print("|x_N| <= ", xN_bound_ei(g, L, Delta, N))
