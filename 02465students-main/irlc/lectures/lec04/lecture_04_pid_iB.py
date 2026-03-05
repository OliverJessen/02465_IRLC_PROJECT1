# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.lec04.lecture_04_pid_p import pidplot

if __name__ == "__main__":
    pidplot(Kp=40, Kd=50, Ki=10, slope=2, target=0)
