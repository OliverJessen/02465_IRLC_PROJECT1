        self.q_star += self.reward_change_std * np.random.randn(self.k)
        self.optimal_action = np.argmax(self.q_star) 