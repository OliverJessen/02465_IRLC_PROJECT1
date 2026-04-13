        self.episode.append((s, a, r))
        if done:
            returns = get_MC_return_SA(self.episode, self.gamma, self.first_visit)
            for s, a, G in returns:
                # s,a = sa
                if self.alpha is None:
                    self.returns_sum_S[s, a] += G
                    self.returns_count_N[s, a] += 1
                    self.Q[s, a] = self.returns_sum_S[s, a] / self.returns_count_N[s, a]
                else:
                    self.Q[s, a] += self.alpha * (G - self.Q[s, a])
            self.episode = [] 