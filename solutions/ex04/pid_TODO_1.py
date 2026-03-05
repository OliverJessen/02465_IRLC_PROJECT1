        e = self.target - x 
        # if self.e_prior == 0 and self.I == 0:
        #     self.e_prior = e
        self.I = self.I + e * self.dt
        u = self.Kp * e + self.Ki * self.I + self.Kd * (e - self.e_prior)/self.dt
        self.e_prior = e 