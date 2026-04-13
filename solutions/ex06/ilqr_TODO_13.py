    gs = [model.cost.c(x, u, i, compute_gradients=True) for i, (x, u) in enumerate(zip(x_bar[:-1], u_bar))] 
    c, c_x, c_u, c_xx, c_ux, c_uu = zip(*gs) 