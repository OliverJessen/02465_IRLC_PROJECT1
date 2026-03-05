    model = FlowerStoreModel(N=N, prob_empty=True) 
    J, pi = DP_stochastic(model)
    pr_empty = -J[0][x0] 