# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""

References:
  [Her25] Tue Herlau. Sequential decision making. (Freely available online), 2025.
"""
import numpy as np
import matplotlib.pyplot as plt
from irlc import bmatrix
from irlc import savepdf



def LQR(A : list,  # Dynamic
        B : list,  # Dynamics
        d : list =None,  # Dynamics (optional)
        Q : list=None,
        R: list=None,
        H : list=None,
        q : list=None,
        r : list=None,
        qc : list=None,
        QN : np.ndarray =None, # Terminal cost term
        qN : np.ndarray=None,  # Terminal cost term
        qcN : np.ndarray =None,  # Terminal cost term.
        mu : float =0 # regularization parameter which will only be relevant next week.
        ):
    r"""
    Implement the LQR as defined in (Her25, Algorithm 22). I recommend viewing this documentation online (documentation for week 6).

    When you solve this exercise, look at the algorithm in the book. Since the LQR problem is on the form:

    .. math::

        x_{k+1} = A_k x_k + B_k u_k + d_k

    For :math:`k=0,\dots,N-1` this means there are :math:`N` matrices :math:`A_k`. This is implemented by assuming that
    :python:`A` (i.e., the input argument) is a :python:`list` of length :math:`N` so that :python:`A[k]` corresponds
    to :math:`A_k`.

    Similar conventions are used for the cost term (please see the lecture notes or the online documentation for their meaning). Recall it has the form:

    .. math::

        c(x_k, u_k) = \frac{1}{2} \mathbf x_k^\top Q_k \mathbf x_k + \frac{1}{2} \mathbf q_k^\top \mathbf x_k + q_k + \cdots

    When the function is called, the vector :math:`\textbf{q}_k` corresponds to :python:`q` and the constant :math:`q_k` correspond to :python:`qc` (q-constant).

    .. note::

        Only the terms :python:`A` and :python:`B` are required. The rest of the terms will default to 0-matrices.

    The LQR algorithm will ultimately compute a control law of the form:

    .. math::

        \mathbf u_k = L_k \mathbf x_k + \mathbf l_k

    And a cost-to-go function as:

    .. math::

        J_k(x_k) = \frac{1}{2} \mathbf x_k^\top V_k \mathbf x_k + v_k^\top \mathbf x_k + v_k

    Again there are :math:`N-1` terms. The function then return :python:`return (L, l), (V, v, vc)` so that :python:`L[k]` corresponds to :math:`L_k`.

    :param A: A list of :python:`np.ndarray` containing all terms :math:`A_k`
    :param B: A list of :python:`np.ndarray` containing all terms :math:`B_k`
    :param d: A list of :python:`np.ndarray` containing all terms  :math:`\mathbf d_k` (optional)
    :param Q: A list of :python:`np.ndarray` containing all terms  :math:`Q_k` (optional)
    :param R: A list of :python:`np.ndarray` containing all terms  :math:`R_k` (optional)
    :param H: A list of :python:`np.ndarray` containing all terms  :math:`H_k` (optional)
    :param q: A list of :python:`np.ndarray` containing all terms  :math:`\mathbf q_k` (optional)
    :param r: A list of :python:`np.ndarray` containing all terms  :math:`\mathbf r_k` (optional)
    :param qc: A list of :python:`float` containing all terms  :math:`q_k` (i.e., constant terms) (optional)
    :param QN: A :python:`np.ndarray` containing the terminal cost term :math:`Q_N` (optional)
    :param qN: A :python:`np.ndarray` containing the terminal cost term :math:`\mathbf q_N` (optional)
    :param qcN: A :python:`np.ndarray` containing the terminal cost term :math:`q_N`
    :param mu: A regularization term which is useful for iterative-LQR (next week). Default to 0.
    :return: A tuple of the form :python:`(L, l), (V, v, vc)` corresponding to the control and cost-matrices.
    """
    N = len(A)
    n,m = B[0].shape
    # Initialize empty lists for control matrices and cost terms
    L, l = [None]*N, [None]*N
    V, v, vc = [None]*(N+1), [None]*(N+1), [None]*(N+1)
    # Initialize constant cost-function terms to zero if not specified.
    # They will be initialized to zero, meaning they have no effect on the update rules.
    QN = np.zeros((n,n)) if QN is None else QN
    qN = np.zeros((n,)) if qN is None else qN
    qcN = 0 if qcN is None else qcN
    Q = init_mat(Q, n, n, N=N)
    R = init_mat(R, m, m, N=N)
    H, q, qc, r = init_mat(H,m,n,N=N), init_mat(q,n,N=N), init_mat(qc,1,N=N), init_mat(r,m,N=N)
    d = init_mat(d,n, N=N)
    """ In the next line, you should initialize the last cost-term. This is similar to how we in DP had the initialization step
    > J_N(x_N) = g_N(x_N)
    Except that since x_N is no longer discrete, we store it as matrices/vectors representing a second-order polynomial, i.e.    
    > J_N(X_N) = 1/2 * x_N' V[N] x_N + v[N]' x_N + vc[N]
    """
    V[N], v[N], vc[N] = QN, qN, qcN

    In = np.eye(n)
    for k in range(N-1,-1,-1):
        # When you update S_uu and S_ux remember to add regularization as the terms ... (V[k+1] + mu * In) ...
        # Note that that to find x such that
        # >>> x = A^{-1} y this
        # in a numerically stable manner this should be done as
        # >>> x = np.linalg.solve(A, y)
        # The terms you need to update will be, in turn:
        # Suu = ...
        # Sux = ...
        # Su = ...
        # L[k] = ...
        # l[k] = ...
        # V[k] = ...
        # V[k] = ...
        # v[k] = ...
        # vc[k] = ...
        ## TODO: Half of each line of code in the following 4 lines have been replaced by garbage. Make it work and remove the error.
        #----------------------------------------------------------------------------------------------------------------------------
        Suu = R[k] + B[k].T @ (V[k+1] + mu * In) @ B[k]
        Sux = H[k] + B[k].T @ (V[k+1] + mu * In) @ A[k]
        Su = r[k] + B[k].T @ (v[k + 1] + V[k + 1] @ d[k])
        L[k] = -np.linalg.solve(Suu, Sux)
        l[k] = -np.linalg.solve(Suu, Su) # You get this for free. Notice how we use np.lingalg.solve(A,x) to compute A^{-1} x
        V[k] = Q[k] + A[k].T @ V[k+1] @ A[k] - L[k].T @ Suu @ L[k]
        V[k] = 0.5 * (V[k] + V[k].T)  # I recommend putting this here to keep V positive semidefinite
        # You get these for free: Compare to the code in the algorithm.
        v[k] = q[k] + A[k].T @ (v[k+1]  + V[k+1] @ d[k]) + Sux.T @ l[k]
        vc[k] = vc[k+1] + qc[k] + d[k].T @ v[k+1] + 1/2*( d[k].T @ V[k+1] @ d[k] ) + 1/2*l[k].T @ Su

    return (L,l), (V,v,vc)


def init_mat(X, a, b=None, N=None):
    """
    Helper function. Check if X is None, and if so return a list
      [A, A,....]
    which is N long and where each A is a (a x b) zero-matrix, else returns X repeated N times:
     [X, X, ...]
    """
    M0 = np.zeros((a,) if b is None else (a, b))
    if X is not None:
        return [m if m is not None else M0 for m in X]
    else:
        return [M0] * N

def lqr_rollout(x0,A,B,d,L,l):
    """
    Compute a rollout (states and actions) given solution from LQR controller function.

    x0 is a vector (starting state), and A, B, d and L, l are lists of system/control matrices.
    """
    x, states,actions = x0, [x0], []
    n,m = B[0].shape
    N = len(L)
    d = init_mat(d,n,1,N)  # Initialize as a list of zero matrices [ np.zeros((n,1)), np.zeros((n,1)), ...]
    l = init_mat(l,m,1,N)  # Initialize as a list of zero matrices [ np.zeros((m,1)), np.zeros((m,1)), ...]

    for k in range(N):
        u = L[k] @ x + l[k]
        x = A[k] @ x + B[k] @ u + d[k]
        actions.append(u)
        states.append(x)
    return states, actions

if __name__ ==  "__main__":
    """
    Solve this problem (see also lecture notes for the same example)
    http://cse.lab.imtlucca.it/~bemporad/teaching/ac/pdf/AC2-04-LQR-Kalman.pdf
    """
    N = 20
    A = np.ones((2,2))
    A[1,0] = 0
    B = np.asarray([[0], [1]])
    Q = np.zeros((2,2))
    R = np.ones((1,1))

    print("System matrices A, B, Q, R")
    print(bmatrix(A))  
    print(bmatrix(B))  
    print(bmatrix(Q))  
    print(bmatrix(R))  

    for rho in [0.1, 10, 100]:
        Q[0,0] = 1/rho
        (L,l), (V,v,vc) = LQR(A=[A]*N, B=[B]*N, d=None, Q=[Q]*N, R=[R]*N, QN=Q)

        x0 = np.asarray( [[1],[0]])
        trajectory, actions = lqr_rollout(x0,A=[A]*N, B=[B]*N, d=None,L=L,l=l)

        xs = np.concatenate(trajectory, axis=1)[0,:]

        plt.plot(xs, 'o-', label=f'rho={rho}')

        k = 10
        print(f"Control matrix in u_k = L_k x_k + l_k at k={k}:", L[k])
    for k in [N-1,N-2,0]:
        print(f"L[{k}] is:", L[k].round(4))
    plt.title("Double integrator")
    plt.xlabel('Steps $k$')
    plt.ylabel('$x_1 = $ x[0]')
    plt.legend()
    plt.grid()
    savepdf("dlqr_double_integrator")
    plt.show()
