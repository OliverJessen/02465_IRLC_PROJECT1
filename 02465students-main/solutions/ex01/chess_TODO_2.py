        if np.random.rand() < self.p_draw: 
            game_outcome = 0
        else:
            if np.random.rand() < self.p_win:
                game_outcome = 1
            else:
                game_outcome = -1 