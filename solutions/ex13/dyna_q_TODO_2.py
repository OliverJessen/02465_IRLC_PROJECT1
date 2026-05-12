            s_, a_, r_, sp_,done_ = self.Model[np.random.randint(len(self.Model))]
            self.q_update(s_,a_,r_,sp_,done_, info_s, info_sp) 