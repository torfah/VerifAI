import carla
from agents.navigation.agent import *
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import distance_vehicle, get_speed

import numpy as np

class PIDsafeController():

    def __init__(self, vehicle, target_speed, opt_dict=None):

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
        self.target_speed = target_speed 
        self.lateral_pid_dict['dt'] = 1.0 / self.target_speed
        self.longitudinal_pid_dict['dt'] = 1.0 / self.target_speed
        if opt_dict:
            if 'lateral_pid_dict' in opt_dict:
                self.lateral_pid_dict = opt_dict['lateral_pid_dict']
            if 'longitudinal_pid_dict' in opt_dict:
                self.longitudinal_pid_dict = opt_dict['longitudinal_pid_dict']
        self.controller = VehiclePIDController(vehicle,
                                              args_lateral=self.lateral_pid_dict,
                                              args_longitudinal=self.longitudinal_pid_dict)
        self.has_stopped = False
        self._vehicle = vehicle
    def _calc_dtc(self, vehicle, waypoint):
        v_yaw = vehicle.get_transform().rotation.yaw
        w_yaw = waypoint.transform.rotation.yaw
        dtc = math.sin(math.radians( abs(v_yaw - w_yaw) )) * distance_vehicle(waypoint, vehicle.get_transform())
        return dtc
    def run_step(self, waypoint):

        current_speed = get_speed(self._vehicle) 
        dtc = self._calc_dtc(self._vehicle, waypoint)
        isBack2Center = False
        control = self.controller.run_step(self.target_speed, waypoint)
        if not self.has_stopped: 
            if current_speed < 0.05:
                self.has_stopped = True
            else:
                control.throttle = 0.0
                control.brake = 1.0
                print ("brake")
        else: 
            if dtc < 0.2:
                self.has_stopped = False
                isBack2Center = True
        
        return control, isBack2Center
