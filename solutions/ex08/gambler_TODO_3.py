        r = 1 if s + a == 100 else 0
        WIN = (s+a, r)
        LOSS = (s-a, 0)
        outcome_dict = {WIN: self.p_heads, LOSS: 1-self.p_heads } if WIN != LOSS else {WIN: 1.} 