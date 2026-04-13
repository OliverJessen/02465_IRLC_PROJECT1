    # fs = [(v[1],v[2]) for v in [model.f(x, u, k, compute_jacobian=True) for k, (x, u) in enumerate(zip(x_bar[:-1], u_bar))]]  
    fs = [model.f_jacobian(x, u, k) for k, (x, u) in enumerate(zip(x_bar[:-1], u_bar))]

    A, B = zip(*fs) 