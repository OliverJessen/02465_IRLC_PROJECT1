        if isinstance(s, np.ndarray):
            print("Bad type.")
        self.v[s] += self.alpha * (r + self.gamma * (self.v[sp] if not done else 0) - self.v[s]) 