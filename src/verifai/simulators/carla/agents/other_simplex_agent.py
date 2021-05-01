import carla
from agents.navigation.agent import *
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import distance_vehicle, draw_waypoints, get_speed

from verifai.simulators.carla.agents.pid_advanced_controller import *
from verifai.simulators.carla.agents.pid_safe_controller import *
import numpy as np
import sys
from examples.carla.overtake_control.config import *
import examples.carla.overtake_control.simpath.monitor as simplex_monitor
'''Agent that follows road waypoints (prioritizing a straight
trajectory if multiple options available) using longitudinal
and lateral PID.'''
def d_angle(a, b):
    return abs((a - b + 180) % 360 - 180)
class OtherSimplexAgent(Agent):

    def __init__(self, vehicle, opt_dict=None):
        super(OtherSimplexAgent, self).__init__(vehicle)
        safe_speed = 10.0
        if opt_dict:
            if 'target_speed' in opt_dict:
                self.target_speed = opt_dict['target_speed']
        self.safe_controller = PIDsafeController(vehicle, safe_speed, opt_dict)
        self.advanced_controller = PIDadvancedController(vehicle, opt_dict)
        self.features = {}
        self.opt_dict = opt_dict

        self.waypoints = []
        self.safe_waypoints = []
        self.max_waypoints = 200 

        self.radius = self.target_speed / 50 
        self.min_dist = 0.9 * self.radius 
        self.isBack2Center =True 
    def add_next_waypoints(self, waypoints, radius):
        if not waypoints:
            current_w = self._map.get_waypoint(self._vehicle.get_location())
            waypoints.append(current_w)
        while len(waypoints) < self.max_waypoints:
            last_w = waypoints[-1]
            last_heading = last_w.transform.rotation.yaw
            next_w_list = list(last_w.next(self.radius))
            if next_w_list:
                next_w  = max(next_w_list,
                              key = lambda w: d_angle(w.transform.rotation.yaw, last_heading))
            else:
                print('No more waypoints.')
                return
            waypoints.append(next_w)

    def run_step(self):
        transform = self._vehicle.get_transform()

        if self.waypoints:
            # If too far off course, reset waypoint queue.
            if distance_vehicle(self.waypoints[0], transform) > 5.0 * self.radius:
                self.waypoints = []

        # Get more waypoints.
        if len(self.waypoints) < self.max_waypoints // 2:
            self.add_next_waypoints(self.waypoints, self.radius)

        # If no more waypoints, stop.
        if not self.waypoints:
            print('Ran out of waypoints; stopping.')
            control = carla.VehicleControl()
            control.throttle = 0.0
            return control


        # Remove waypoints we've reached.
        while distance_vehicle(self.waypoints[0], transform) < self.min_dist:
            self.waypoints = self.waypoints[1:]

        # Draw next waypoint
        draw_waypoints(self._vehicle.get_world(),
                           self.waypoints[:1],
                           self._vehicle.get_location().z + 1.0)

        dtc = self.get_features_and_return_dtc()
        do_AC = simplex_monitor.check(self.features, INPUT_WINDOW, False) 
        if do_AC and self.isBack2Center:
            v_yaw = self._vehicle.get_transform().rotation.yaw
            yaw_diff8 = d_angle(self.waypoints[8].transform.rotation.yaw, v_yaw)
            yaw_diff0 = d_angle(self.waypoints[0].transform.rotation.yaw, v_yaw)
            
            control = self.advanced_controller.run_step(self.waypoints[0], yaw_diff0, yaw_diff8)
        else:
            #if not self.safe_waypoints:
            #    self.safe_waypoints = self.waypoints[:1]
            #if len(self.safe_waypoints) < self.max_waypoints // 2:
            #    self.add_next_waypoints(self.safe_waypoints, self.radius/3)
            #while distance_vehicle(self.safe_waypoints[0], transform) < self.min_dist:
            #    self.safe_waypoints = self.safe_waypoints[1:]
            control, self.isBack2Center = self.safe_controller.run_step(self.waypoints[0], dtc)
            #if self.isBack2Center:
            #    self.safe_waypoints = []
        return control 

    def get_features_and_return_dtc(self):
        get_scalar = lambda vec: 3.6 * math.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)

        v_yaw = self._vehicle.get_transform().rotation.yaw
        self.features['v'] = get_scalar(self._vehicle.get_velocity())
        #self.features['acc'] = get_scalar(self._vehicle.get_acceleration())
        #self.features['ang_v'] = get_scalar(self._vehicle.get_angular_velocity())
        dtc = 0
        for i in range(5, -1, -1):
            waypoint = self.waypoints[i]
            w_yaw = waypoint.transform.rotation.yaw 

            diff_yaw = abs(math.tan(math.radians( d_angle(v_yaw , w_yaw) )))
            distance = distance_vehicle(waypoint, self._vehicle.get_transform())
            dtc = diff_yaw * distance 
            #self.features[f'waypoint_{i}_dyaw'] = diff_yaw
            #self.features[f'waypoint_{i}_dist'] = distance
            self.features[f'waypoint_{i}_dtc'] = dtc
        
        return dtc
