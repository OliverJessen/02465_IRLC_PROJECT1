    def pi(self,x, k, info=None): 
        action = self.L[0] @ x + self.l[0]
        return action 