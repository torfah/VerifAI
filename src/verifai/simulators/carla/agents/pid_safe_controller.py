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
        self.old_target_speed = target_speed
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
    def run_step(self, waypoint, dtc):

        current_speed = get_speed(self._vehicle) 

        isBack2Center = False
        control = self.controller.run_step(self.target_speed, waypoint)
        if not self.has_stopped: 
            if current_speed < 0.5:
                self.has_stopped = True
                self.target_speed = 20.0
            else:
                control.throttle = 0.0
                control.brake = 0.5
        else: 
            if dtc < 0.5:
                self.has_stopped = False
                isBack2Center = True
                self.target_speed = self.old_target_speed 

        # TODO: brake if there is a car in front
        '''
        vehicles = self._world.get_actors().filter('*vehicle*')
        other_forward_v = None
        for v in vehicles:
            # Check if v is self.
            if v.id != self._vehicle.id:
                other_vt = v.get_transform()
        if (other_vt != None):
            self_vt = self._vehicle.get_transform()
            dis_vec = other_vt.location - self_vt.location
            relative_dis = np.sqrt(dis_vec.x ** 2 + dis_vec.y ** 2 + dis_vec.z ** 2)
            if relative_dis <= 5:
                control.throttle = 0.0
                control.brake = 0.8
        '''


        return control, isBack2Center

