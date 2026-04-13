                    self.returns_sum_S[s] += G
                    self.returns_count_N[s] += 1.0
                    self.v[s] = self.returns_sum_S[s] / self.returns_count_N[s] 