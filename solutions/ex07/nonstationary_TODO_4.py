    bandit = NonstationaryBandit(k=10) 

    agents = [BasicAgent(bandit, epsilon=epsilon)]
    agents += [MovingAverageAgent(bandit, epsilon=epsilon, alpha=alpha) for alpha in alphas] 