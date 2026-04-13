        A, B, c, c_x, c_u, c_xx, c_ux, c_uu = get_derivatives(model, x_bar, u_bar) 
        J = sum(c) 