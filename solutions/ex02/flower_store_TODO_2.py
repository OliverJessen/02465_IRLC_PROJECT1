    model = FlowerStoreModel(N=N, c=c, prob_empty=False) 
    J, pi = DP_stochastic(model)
    u = pi[0][x0]  