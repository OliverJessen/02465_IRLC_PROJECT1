        Delta = tt[k + 1] - tt[k] 
        xn = xs[k]
        k1 = np.asarray(f(xn, u))
        k2 = np.asarray(f(xn + Delta * k1/2, u))
        k3 = np.asarray(f(xn + Delta * k2/2, u))
        k4 = np.asarray(f(xn + Delta * k3,   u))
        x_next = xn + 1/6 * Delta * (k1 + 2*k2 + 2*k3 + k4) 