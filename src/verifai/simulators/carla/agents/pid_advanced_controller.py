import carla
from agents.navigation.agent import *
from agents.navigation.controller import VehiclePIDController

import numpy as np

class PIDadvancedController():

    def __init__(self, vehicle, opt_dict=None):

        # Default params:
        self.target_speed = 20.0

        # Default PID params:
        self.lateral_pid_dict = {
            'K_P': 1.0,
            'K_D': 0.01,
            'K_I': 0.6,
            'dt': 0.05
        }
        self.longitudinal_pid_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 1,
            'dt': 0.05
        }
        if opt_dict:
            if 'target_speed' in opt_dict:
                self.target_speed = opt_dict['target_speed']
                self.lateral_pid_dict['dt'] = 1.0 / self.target_speed
                self.longitudinal_pid_dict['dt'] = 1.0 / self.target_speed
            if 'lateral_pid_dict' in opt_dict:
                self.lateral_pid_dict = opt_dict['lateral_pid_dict']
            if 'longitudinal_pid_dict' in opt_dict:
                self.longitudinal_pid_dict = opt_dict['longitudinal_pid_dict']
        self.controller = VehiclePIDController(vehicle,
                                              args_lateral=self.lateral_pid_dict,
                                              args_longitudinal=self.longitudinal_pid_dict)

    def run_step(self, waypoint, yaw_diff0, yaw_diff8):
        coef =1.0
        if yaw_diff0 > 60:
            coef = 0.5
        elif yaw_diff8 > 60:
            coef = 0.8
        speed = self.target_speed * coef
        #print (f"yaw_diff0 {yaw_diff0} yaw_diff8 {yaw_diff8} coef {coef}")
        return self.controller.run_step(speed, waypoint)

