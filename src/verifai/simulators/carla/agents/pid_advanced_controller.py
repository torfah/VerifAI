import carla
from agents.navigation.agent import *
from agents.navigation.controller import VehiclePIDController
import random
import numpy as np
from collections import deque
import math

class PIDadvancedController():

    def __init__(self, vehicle, opt_dict=None):

        # Default params:
        self.target_speed = 20.0
        self.prev_target_speed = self.target_speed
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
        self.cruise_controller = PIDAdaptiveCruiseController(vehicle)

    def run_step(self, waypoint, yaw_diff0, yaw_diff8):
        coef =1.0
        thresh = random.uniform(60, 135)
        if yaw_diff0 > thresh: 
            coef = random.uniform(0.5,0.7)
        elif yaw_diff8 > thresh: 
            coef = random.uniform(0.7,0.9)
        #speed = self.target_speed * coef
        speed = self.cruise_controller.run_step(target_dist=30, target_speed=self.target_speed,
                                                prev_setpoint=self.prev_target_speed, debug=True)
        self.prev_target_speed = speed
        return self.controller.run_step(speed, waypoint)

class PIDAdaptiveCruiseController():
    """
    PIDAdaptiveCruiseController adjust a target speed based on the distance to a car in front
    to feed into a VehiclePIDController for lateral and longitudinal control
    """


    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_dist, target_speed, prev_setpoint, debug=False):
        """
        Execute one step of control to reach a given distance to the car in front.

            :param target_dist: target distance in m
            :param target_speed: target speed on an open road (no car in front)
            :param prev_setpoint: teh previous setpoint for vehicle speed in km/h
            :param debug: boolean for debugging
            :return: target_speed
        """

        # Calculate the distance between the ego car and the closest car in front. 
        # Set current_dist = None if there are no cars nearby in front
        current_dist = None;
        vehicles = self._world.get_actors().filter('*vehicle*')
        other_vl = [v.get_location() for v in vehicles if v.id != self._vehicle.id]
        self_vl = self._vehicle.get_location()
        self_fvec = self._vehicle.get_transform().rotation.get_forward_vector()
        self_fvec = np.array([self_fvec.x, self_fvec.y, self_fvec.z])
        if (other_vl):
            # Find the displacement and distances of all other vehicles in the world relative to the ego car
            disp_vecs = [loc - self_vl for loc in other_vl]
            dists = [self_vl.distance(dv) for dv in disp_vecs]
            disp_vecs = [[v.x, x.y, v.z] for v in disp_vecs]
            loc_info = list(zip(disp_vecs, dists))
            filtered_dists = []
            for disp, dist in loc_info:
                # Compute angle between car heading and other vehicle

                _dot = math.acos(np.clip(np.dot(self_fvec, disp) /
                             (np.linalg.norm(self_fvec) * np.linalg.norm(disp)), -1.0, 1.0))
                _cross = np.cross(disp, self_fvec)
                if _cross[2] < 0:
                    _dot *= -1.0
                if debug:
                    print('Computed angle = {}'.format(_dot))
                # If the other vehicle is roughly within pi/3 of the car orientation, consider it "in front"
                if 0 < _dot < 1:
                    filtered_dists.append(dist)
            # Return the closest distance to anything in front of the car
            current_dist = min(filtered_dists)

        if debug:
            print('Current closest dist = {}'.format(current_dist))

        # TODO: take outpout of pid control and smooth pasted on previous speed setpoint before returning
        if current_dist is not None:
            new_setpoint = self._pid_control(target_dist, current_dist)
        else:
            new_setpoint = target_speed
        return new_setpoint

    def _pid_control(self, target_dist, current_dist):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_dist:  target distance to next car in m
            :param current_dist: current distance to next car in m
            :return: speed control setpoint
        """

        error = target_dist - current_dist
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)
