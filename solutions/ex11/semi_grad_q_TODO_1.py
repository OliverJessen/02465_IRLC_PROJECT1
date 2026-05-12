        if not done:
            a_star = self.Q.get_optimal_action(sp, info_sp)
        td_delta = r + (0 if done else self.gamma * self.Q(sp, a_star)) - self.Q(s, a)
        self.Q.w += self.alpha * td_delta * self.Q.x(s, a) 