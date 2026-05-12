        def train_(Q1,Q2, s, a, r, sp, done=False):
            Q1[s,a] += self.alpha * (r + (self.gamma *  Q2[sp,Q1.get_optimal_action(sp, info_sp)] if not done else 0) - Q1[s,a] )

        train_(self.Q1, self.Q2, s, a, r, sp,done) if np.random.rand() < 0.5 else train_(self.Q2, self.Q1, s, a, r, sp,done) 