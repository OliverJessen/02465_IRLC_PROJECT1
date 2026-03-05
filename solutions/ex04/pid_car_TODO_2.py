        xx = x[5] + x[3] if self.use_both_x5_x3 else x[5] 
        u = np.asarray([self.pid_angle.pi(xx), self.pid_velocity.pi(x[0])]) 