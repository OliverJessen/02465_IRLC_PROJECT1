        self.pid_angle = PID(dt=env.discrete_model.dt, Kp=1.0, Ki=0, Kd=0, target=0) 
        self.pid_velocity = PID(dt=env.discrete_model.dt, Kp=1.5, Ki=0, Kd=0, target=v_target) 