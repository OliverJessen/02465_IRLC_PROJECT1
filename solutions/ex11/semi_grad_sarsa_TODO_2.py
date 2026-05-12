        a_prime = super().pi(sp, k=0, info=info_sp) 
        delta = r + (0 if done else self.gamma * self.Q(sp, a_prime)) - self.Q(s, a)
        self.Q.w += self.alpha * delta * self.Q.x(s,a)
        self.a = a_prime 