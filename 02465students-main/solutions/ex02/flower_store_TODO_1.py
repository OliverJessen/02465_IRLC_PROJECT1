class FlowerStoreModel(InventoryDPModel): 
    def __init__(self, N=3, c=0., prob_empty=False):
        self.c = c
        self.prob_empty = prob_empty
        super().__init__(N=N)

    def g(self, x, u, w, k):  # Cost function g_k(x,u,w)
        if self.prob_empty:
            return 0
        return u * self.c + np.abs(x + u - w)

    def f(self, x, u, w, k):  # Dynamics f_k(x,u,w)
        return max(0, min(max(self.S(k)), x + u - w))

    def Pw(self, x, u, k):  # Distribution over random disturbances
        pw = {0: .1, 1: .3, 2: .6}
        return pw

    def gN(self, x):
        if self.prob_empty:
            return -1 if x == 1 else 0
        else:
            return 0 