        sp[done, :] = 0 
        astar = np.argmax(self.Q(sp), axis=1)  * (1-np.asarray(done))
        y = r[:,0] + self.gamma * self.target(sp)[range(len(sp)), astar] * (1 - done)
        target = self.Q(s)
        target[range(len(a)), a] = y 