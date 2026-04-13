        pi_a = self.Pa()
        for b in range(self.k):
            if b == a:
                self.H[b] += self.alpha * (r - self.R_bar) * (1 - pi_a[b])
            else:
                self.H[b] -= self.alpha * (r - self.R_bar) * pi_a[b]

        if self.baseline:
            self.R_bar = self.R_bar + (self.alpha if self.alpha is not None else 1/(self.t+1)) * (r - self.R_bar) 