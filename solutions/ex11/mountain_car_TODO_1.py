    for i, alpha in enumerate(alphas): 
        n = n_steps[i]
        agent = LinearSemiGradSarsaN(env, gamma=1, alpha=alpha / num_of_tilings, epsilon=0, n=n)
        experiment = f"experiments/mountaincar_10-2_{agent}_{episodes}"
        train(env, agent, experiment_name=experiment, num_episodes=episodes, max_runs=max_runs)
        experiments.append(experiment)

    agent = LinearSemiGradSarsaLambda(env, gamma=1, alpha=alphas[1]/num_of_tilings, epsilon=0, lamb=0.9)
    experiment = f"experiments/mountaincar_10-2_{agent}_{episodes}"
    train(env, agent, experiment_name=experiment, num_episodes=episodes, max_runs=max_runs)
    experiments.append(experiment)

    agent = LinearSemiGradQAgent(env, gamma=1, alpha=alphas[1] / num_of_tilings, epsilon=0)
    experiment = f"experiments/mountaincar_10-2_{agent}_{episodes}"
    train(env, agent, experiment_name=experiment, num_episodes=episodes, max_runs=max_runs)
    experiments.append(experiment) 