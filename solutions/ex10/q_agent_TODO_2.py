        if not done:
            a_star = self.Q.get_optimal_action(sp, info_sp)
        self.Q[s,a] += self.alpha * (r + self.gamma * (0 if done else self.Q[sp,a_star]) - self.Q[s,a]) 