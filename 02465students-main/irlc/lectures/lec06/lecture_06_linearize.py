# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex06.linearization_agent import get_offbalance_cart

if __name__ == "__main__":
    env = get_offbalance_cart(waiting_steps=20, sleep_time=0.1)
    env.close()
