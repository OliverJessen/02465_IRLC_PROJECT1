    toy = Toy2DControl() 
    x0 = np.asarray([np.pi/2, 0])
    xs, us, ts, cost = toy.simulate( x0=x0, u_fun = u0, t0=0, tF=T)
    wT = xs[-1][0] 