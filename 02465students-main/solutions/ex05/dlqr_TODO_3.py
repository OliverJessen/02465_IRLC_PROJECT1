        Suu = R[k] + B[k].T @ (V[k+1] + mu * In) @ B[k]  
        Sux = H[k] + B[k].T @ (V[k+1] + mu * In) @ A[k]
        Su = r[k] + B[k].T @ v[k + 1]  + B[k].T @ V[k + 1] @ d[k]
        L[k] = -np.linalg.solve(Suu, Sux) 