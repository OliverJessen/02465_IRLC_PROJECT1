        terminated = True  
        if a == 0:
            s_next = self.s * 1.1
        else:
            if np.random.rand() < 1/4:
                s_next = 0
            else:
                s_next = self.s + 12
        reward = s_next - self.s  