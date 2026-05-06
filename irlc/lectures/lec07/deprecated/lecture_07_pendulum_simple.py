# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex04.model_pendulum import GymSinCosPendulumEnvironment
from irlc.utils.video_monitor import VideoMonitor
from irlc.ex04.discrete_control_cost import goal_seeking_qr_cost, DiscreteQRCost
from irlc.ex01.agent import train
from irlc.ex06.lqr_learning_agents import MPCLocalLearningLQRAgent, MPCLearningAgent
from irlc import plot_trajectory, main_plot
import matplotlib.pyplot as plt
import numpy as np
from irlc.ex06.mpc_pendulum_experiment_lqr import mk_mpc_pendulum_env

L = 12
def main_pendulum_lqr_simple(Tmax=10):

    """ Run Local LQR/MPC agent using the parameters
    L = 12  
    neighboorhood_size = 50
    min_buffer_size = 50 
    """
    env_pendulum = mk_mpc_pendulum_env()

    # agent = .... (instantiate agent here)
    # TODO: 1 lines missing.
    raise NotImplementedError("Instantiate your agent here")
    env_pendulum = VideoMonitor(env_pendulum)

    experiment_name = f"pendulum{L}_lqr"
    stats, trajectories = train(env_pendulum, agent, experiment_name=experiment_name, num_episodes=16,return_trajectory=True)
    plt.show()
    for k in range(len(trajectories)):
        plot_trajectory(trajectories[k], env_pendulum)
        plt.title(f"Trajectory {k}")
        plt.show()

    env_pendulum.close()
    main_plot(experiment_name)
    plt.show()

if __name__ == "__main__":
    np.random.seed(1)
    main_pendulum_lqr_simple()
