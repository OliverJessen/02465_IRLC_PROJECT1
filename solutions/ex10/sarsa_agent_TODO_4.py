        delta = r + (self.gamma * self.Q[sp,self.a] if not done else 0) - self.Q[s,a] 
        self.Q[s,a] += self.alpha * delta 