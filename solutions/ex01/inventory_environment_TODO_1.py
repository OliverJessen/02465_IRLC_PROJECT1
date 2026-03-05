        s_next = max(0, min(2, self.s-w+a))           # next state; x_{k+1} =  f_k(x_k, u_k, w_k) 
        reward = -(a + (self.s + a - w)**2)           # reward = -cost      = -g_k(x_k, u_k, w_k)
        terminated = self.k == self.N-1               # Have we terminated? (i.e. is k==N-1)
        self.s = s_next                               # update environment state
        self.k += 1                                   # update current time step 