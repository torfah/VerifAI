import carla
from agents.navigation.agent import *
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import distance_vehicle, draw_waypoints, get_speed

import numpy as np

'''Agent that follows road waypoints (prioritizing a straight
trajectory if multiple options available) using longitudinal
and lateral PID.'''
class PIDsafeAgent(Agent):

    def __init__(self, vehicle, opt_dict=None):
        super(PIDsafeAgent, self).__init__(vehicle)

        # Default agent params:
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
            if 'radius' in opt_dict:
                self.radius = opt_dict['radius']
            if 'min_dist' in opt_dict:
                self.min_dist = opt_dict['min_dist']
            if 'lateral_pid_dict' in opt_dict:
                self.lateral_pid_dict = opt_dict['lateral_pid_dict']
            if 'longitudinal_pid_dict' in opt_dict:
                self.longitudinal_pid_dict = opt_dict['longitudinal_pid_dict']
        self.controller = VehiclePIDController(vehicle,
                                              args_lateral=self.lateral_pid_dict,
                                              args_longitudinal=self.longitudinal_pid_dict)
        self.waypoints = []
        self.max_waypoints = 200

        self.radius = self.target_speed / 18    # Radius at which next waypoint is sampled
        self.min_dist = 0.9 * self.radius  # If min_dist away from waypoint[0], move on to next one.
        self.has_stopped = False
        self.has_deviated = False
        self.do_stop = False
        self.old_target_speed = self.target_speed
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


    def run_step(self):
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
        control = self.controller.run_step(self.target_speed, self.waypoints[0])

        current_speed = get_speed(self._vehicle) 
        v_loc = self._vehicle.get_transform().location
        v_yaw = self._vehicle.get_transform().rotation.yaw
        w_loc = self.waypoints[0].transform.location
        w_yaw = self.waypoints[0].transform.rotation.yaw
        dtc = math.sin(math.radians( abs(v_yaw - w_yaw) )) * distance_vehicle(self.waypoints[0], self._vehicle.get_transform())
        print ("v_loc", v_loc, "v_yaw", v_yaw, "w_loc", w_loc, "w_yaw", w_yaw, "dtc", dtc) 
        print ("current speed", current_speed, "do stop", self.do_stop, "has stopped", self.has_stopped)
        if dtc >= 1.0: self.has_deviated = True 
        if self.has_deviated and not self.has_stopped: 
            self.target_speed = 5.0
            if current_speed < 0.05:
                self.do_stop = False
                self.has_stopped = True
            else:
                self.do_stop = True
        if dtc < 0.2:
            self.has_stopped = False
            self.has_deviated = False 
            self.target_speed = self.old_target_speed
        if self.do_stop: 
            control.throttle = 0.0
            control.brake = 1.0
            print ("brake")
        
        return control 

