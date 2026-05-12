        Q = self.Q.w @ self.x  
        Q_prime = self.Q.w @ x_prime if not done else None
        delta = r + (self.gamma * Q_prime if not done else 0) - Q
        self.z = self.gamma * self.lamb * self.z + (1-self.alpha * self.gamma * self.lamb *self.z @ self.x) * self.x
        self.Q.w += self.alpha * (delta + Q - self.Q_old) * self.z - self.alpha * (Q-self.Q_old) * self.x  