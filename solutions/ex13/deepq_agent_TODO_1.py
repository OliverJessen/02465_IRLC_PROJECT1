        y = r[:,0] + self.gamma * np.max(self.Q(sp), axis=1) * (1-done) 
        target = self.Q(s)
        target[range(len(a)), a] = y 