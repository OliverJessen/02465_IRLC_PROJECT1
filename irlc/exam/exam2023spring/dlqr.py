"""

References:
  [Her25] Tue Herlau. Sequential decision making. (Freely available online), 2025.
"""
import numpy as np

def fz(X, a, b=None, N=None):
    """
    Helper function. Check if X is None, and if so return a list
      [A, A,....]
    which is N long and where each A is a (a x b) zero-matrix, else returns X repeated N times:
     [X, X, ...]
    """
    if X is not None:
        return X
    X = np.zeros((a,) if b is None else (a,b))
    return [X] * N

def LQR(A, B, d=None, Q=None, R=None, H=None, q=None, r=None, qc=None, QN=None, qN=None, qcN=None, mu=0):
    """
    Implement LQR as defined in (Her25, Algorithm 22). All terms retain their meaning/order of computation from the code; please be careful with the linear algebra!

    Input:
        We follow the convention A, B, etc. are lists of matrices/vectors/scalars, such that
        A_k = A[k] is the dynamical matrix, etc.

        A slight annoyance is we have both a Q-matrix, q-vector and q-constant.
        we will therefore use q to refer to the vector and qc to refer to the scalar.
    Return:
        We will return the (list of) control matrices/vectors L, l such that u = l + L x
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
    H,q,qc,r = fz(H,m,n,N=N), fz(q,n,N=N), fz(qc,1,N=N), fz(r,m,N=N)
    d = fz(d,n, N=N)
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
        Suu = R[k] + B[k].T @ (V[k+1] + mu * In) @ B[k]
        Sux = H[k] + B[k].T @ (V[k+1] + mu * In) @ A[k]
        Su = r[k] + B[k].T @ v[k + 1]  + B[k].T @ V[k + 1] @ d[k]
        L[k] = -np.linalg.solve(Suu, Sux)
        l[k] = -np.linalg.solve(Suu, Su) # You get this for free. Notice how we use np.lingalg.solve(A,x) to compute A^{-1} x
        V[k] = Q[k] + A[k].T @ V[k+1] @ A[k] - L[k].T @ Suu @ L[k]
        V[k] = 0.5 * (V[k] + V[k].T)  # I recommend putting this here to keep V positive semidefinite
        # You get these for free: Compare to the code in the algorithm.
        v[k] = q[k] + A[k].T @ (v[k+1]  + V[k+1] @ d[k]) + Sux.T @ l[k]
        vc[k] = vc[k+1] + qc[k] + d[k].T @ v[k+1] + 1/2*( d[k].T @ V[k+1] @ d[k] ) + 1/2*l[k].T @ Su

    return (L,l), (V,v,vc)
