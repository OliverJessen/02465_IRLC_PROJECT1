                G = sum([self.gamma**(i-tau-1)*self.R[i%(n+1)] for i in range(tau+1, min(tau+n, T)+1)]) 
                S_tau_n, A_tau_n = self.S[(tau+n)%(n+1)], self.A[(tau+n)%(n+1)]
                if tau+n < T:
                    G += self.gamma**n * self._q(S_tau_n, A_tau_n) 