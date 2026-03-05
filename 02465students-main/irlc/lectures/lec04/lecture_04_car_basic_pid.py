# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc import train
from irlc.car.car_model import CarEnvironment
from irlc.ex04.pid_car import PIDCarAgent

if __name__ == "__main__":
    env = CarEnvironment(noise_scale=0, Tmax=30, max_laps=1, render_mode='human')
    agent = PIDCarAgent(env, v_target=.2, use_both_x5_x3=False)
    stats, trajectories = train(env, agent, num_episodes=1, return_trajectory=True)
    env.close()

    # agent = PIDCarAgent(env, v_target=1, use_both_x5_x3=True) # I recommend lowering v_target to make the problem simpler.
