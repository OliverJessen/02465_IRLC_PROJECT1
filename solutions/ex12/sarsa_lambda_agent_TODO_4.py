            self.Q[s,a] += self.alpha * delta * ee 
            self.e[(s,a)] = self.gamma * self.lamb * ee  