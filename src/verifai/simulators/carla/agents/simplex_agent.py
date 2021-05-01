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
class SimplexAgent(Agent):

    def __init__(self, vehicle, opt_dict=None):
        super(SimplexAgent, self).__init__(vehicle)
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
        self.min_dist = 3.0 * self.radius 
        self.isBack2Center =True 
        self.timestamp = 0
        self.buffer=[]
    def add_next_waypoints(self, waypoints, radius):

        if not waypoints:
            current_w = self._map.get_waypoint(self._vehicle.get_location())
            waypoints.append(current_w)
        while len(waypoints) < self.max_waypoints:
            last_w = waypoints[-1]
            last_heading = last_w.transform.rotation.yaw
            next_w_list = list(last_w.next(self.radius))
            # Go straight if possible.
            if next_w_list:
                next_w  = min(next_w_list,
                              key = lambda w: d_angle(w.transform.rotation.yaw, last_heading))
            else:
                print('No more waypoints.')
                return
            waypoints.append(next_w)

    def _write_features(self, iteration):
        with open(f'{SIM_DIR}/{iteration}.log', 'a') as f:
            s = str()
            for key in self.features:
                s+=f'{key} {self.features[key]} '
            s+='\n'
            f.write(s)
    def run_step(self, iteration):
        transform = self._vehicle.get_transform()

        if self.waypoints:
            # If too far off course, reset waypoint queue.
            if distance_vehicle(self.waypoints[0], transform) > 5.0 * self.radius:
                self.waypoints = []

        # Get more waypoints.
        if len(self.waypoints) < self.max_waypoints //2:
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

        other_rdis, other_rheading = self.get_other_car_info()
        dtc = self.get_features_and_return_dtc(other_rdis, other_rheading)
        do_AC = True#simplex_monitor.check(self.features, INPUT_WINDOW, False) 
        if do_AC and self.isBack2Center:
            v_yaw = self._vehicle.get_transform().rotation.yaw
            yaw_diff8 = d_angle(self.waypoints[8].transform.rotation.yaw, v_yaw)
            yaw_diff0 = d_angle(self.waypoints[0].transform.rotation.yaw, v_yaw)
            
            control = self.advanced_controller.run_step(self.waypoints[0], yaw_diff0, yaw_diff8)
        else:
            control, self.isBack2Center = self.safe_controller.run_step(self.waypoints[0], dtc, other_rdis, other_rheading)

            print ("do_SC", dtc)
        self._write_features(iteration)
        self.timestamp += 1
        return control 

    def get_features_and_return_dtc(self, other_rdis, other_rheading):
        get_scalar = lambda vec: 3.6 * math.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)

        v_yaw = self._vehicle.get_transform().rotation.yaw
        self.features['v'] = get_scalar(self._vehicle.get_velocity())

        if other_rdis is not None:
            self.features['other_heading'] = other_rheading
            self.features['other_distance'] = other_rdis

        dtc = 0
        for i in range(5, -1, -1):
            waypoint = self.waypoints[i]
            w_yaw = waypoint.transform.rotation.yaw 

            diff_yaw = abs(math.tan(math.radians( d_angle(v_yaw , w_yaw) )))
            distance = distance_vehicle(waypoint, self._vehicle.get_transform())
            dtc = diff_yaw * distance 
            self.features[f'waypoint_{i}_dtc'] = dtc
        
        return dtc

    def get_other_car_info(self):
        vehicles = self._world.get_actors().filter('*vehicle*')
        other_forward_v = None
        for v in vehicles:
            # Check if v is self.
            if v.id != self._vehicle.id:
                other_vt = v.get_transform()

        if other_vt is None:
            return None, None

        self_vt = self._vehicle.get_transform()

        dis_vec = other_vt.location - self_vt.location
        relative_dis = np.sqrt(dis_vec.x ** 2 + dis_vec.y ** 2 + dis_vec.z ** 2)

        relative_heading = d_angle(other_vt.rotation.yaw , self_vt.rotation.yaw)


        return relative_dis, relative_heading
