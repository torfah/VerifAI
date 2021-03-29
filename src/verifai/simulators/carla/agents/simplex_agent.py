import carla
from agents.navigation.agent import *
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import distance_vehicle, draw_waypoints, get_speed

from verifai.simulators.carla.agents.pid_advanced_controller import *
from verifai.simulators.carla.agents.pid_safe_controller import *
import numpy as np
import sys
'''Agent that follows road waypoints (prioritizing a straight
trajectory if multiple options available) using longitudinal
and lateral PID.'''
class SimplexAgent(Agent):

    def __init__(self, vehicle, opt_dict=None):
        super(SimplexAgent, self).__init__(vehicle)
        safe_speed = 5.0
        if opt_dict:
            if 'target_speed' in opt_dict:
                self.target_speed = opt_dict['target_speed']
        self.safe_controller = PIDsafeController(vehicle, safe_speed, opt_dict)
        self.advanced_controller = PIDadvancedController(vehicle, opt_dict)
        self.features = []
        self.opt_dict = opt_dict

        self.waypoints = []
        self.max_waypoints = 200

        self.radius = self.target_speed / 18  
        self.min_dist = 0.9 * self.radius 
        self.isBack2Center =True 
        self.timestamp = 0
    def add_next_waypoints(self):
        def d_angle(a, b):
            return abs((a - b + 180) % 360 - 180)

        if not self.waypoints:
            current_w = self._map.get_waypoint(self._vehicle.get_location())
            self.waypoints.append(current_w)
        while len(self.waypoints) < self.max_waypoints:
            last_w = self.waypoints[-1]
            last_heading = last_w.transform.rotation.yaw
            next_w_list = list(last_w.next(self.radius))
            # Go straight if possible.
            if next_w_list:
                next_w  = min(next_w_list,
                              key = lambda w: d_angle(w.transform.rotation.yaw, last_heading))
            else:
                print('No more waypoints.')
                return
            self.waypoints.append(next_w)

    def _write_features(self, iteration, dtc):
        with open('/home/carla_challenge/Desktop/wayne/VerifAI/examples/carla/overtake_control/simpath/{}.log'.format(iteration), 'a') as f:
            f.write('time {} dtc {}\n'.format(self.timestamp, dtc))
    def run_step(self, iteration):
        transform = self._vehicle.get_transform()

        if self.waypoints:
            # If too far off course, reset waypoint queue.
            if distance_vehicle(self.waypoints[0], transform) > 5.0 * self.radius:
                self.waypoints = []

        # Get more waypoints.
        if len(self.waypoints) < self.max_waypoints // 2:
            self.add_next_waypoints()

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

        #do_AC = decision_tree(self.features)
        
       # v_loc = self._vehicle.get_transform().location
        v_yaw = self._vehicle.get_transform().rotation.yaw
       # w_loc = waypoints.transform.location
        w_yaw = self.waypoints[0].transform.rotation.yaw #sometimes it gives -270 degrees, which is 90 degrees, add abs to dtc
        dtc = abs( math.sin(math.radians( abs(v_yaw - w_yaw) )) * distance_vehicle(self.waypoints[0], self._vehicle.get_transform()) )
        print ("dtc", dtc)
        self.opt_dict['dtc_history'].append((self.timestamp , dtc))

        do_AC = dtc < 1.0

        if do_AC and self.isBack2Center:
            control = self.advanced_controller.run_step(self.waypoints[0])
        else:
            control, self.isBack2Center = self.safe_controller.run_step(self.waypoints[0], dtc)
        self._write_features(iteration, dtc)
        self.timestamp += 1
        return control 

