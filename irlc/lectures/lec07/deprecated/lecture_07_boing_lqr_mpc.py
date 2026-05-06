# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex04.model_boeing import BoeingEnvironment
from irlc.ex06.lqr_learning_agents import learning_lqr, learning_lqr_mpc, learning_lqr_mpc_local
from irlc.ex06.learning_agent_mpc_optimize import learning_optimization_mpc_local

if __name__ == "__main__":
    env = BoeingEnvironment(output=[10, 0])
    learning_lqr_mpc(env)

    # # Part C: LQR+MPC and local regression
    # learning_lqr_mpc_local(env)
    #
    # # Part D: Optimization+MPC and local regression
    # learning_optimization_mpc_local(env)
