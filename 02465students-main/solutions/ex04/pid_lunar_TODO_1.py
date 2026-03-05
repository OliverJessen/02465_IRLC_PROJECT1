        alt_adj = self.pid_alt.pi( -(np.abs(x[0])- x[1]) ) 
        ang_adj = self.pid_ang.pi( -((.25 * np.pi) * (x[0] + x[2]) - x[4]) ) 