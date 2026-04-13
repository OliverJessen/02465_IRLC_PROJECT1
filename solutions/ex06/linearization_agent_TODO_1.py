        xp = model.f(xbar, ubar, k=0) 
        A, B = model.f_jacobian(xbar, ubar, k=0)

        d = xp - A @ xbar - B @ ubar 