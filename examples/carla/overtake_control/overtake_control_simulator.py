from verifai.simulators.carla.client_carla import *
from verifai.simulators.carla.carla_world import *
from verifai.simulators.carla.carla_task import *

import numpy as np
from dotmap import DotMap
import carla

from verifai.simulators.carla.agents.pid_agent import *
from verifai.simulators.carla.agents.simplex_agent import *
from verifai.simulators.carla.agents.overtake_agent import *
from examples.carla.overtake_control.config import *
# Falsifier (not CARLA) params

def norm(vec):
    return np.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)


class overtake_control_task(carla_task):
    def __init__(self,
                 n_sim_steps=N_SIM_STEP,
                 display_dim=(1280,720),
                 carla_host='127.0.0.1',
                 carla_port=2000,
                 carla_timeout=4.0,
                 world_map='Town02'):
        super().__init__(
            n_sim_steps=n_sim_steps,
            display_dim=display_dim,
            carla_host=carla_host,
            carla_port=carla_port,
            carla_timeout=carla_timeout,
            world_map=world_map
        )


    def use_sample(self, sample):
        print('Sample:', sample)
        iteration = sample[1]
        sample = sample[0]
        self.world.iteration = iteration
        init_conds = sample.init_conditions
        self.ego_target_speed = init_conds.ego_target_speed[0]

        # PID controller parameters
        pid_opt_dict = {
            'target_speed': self.ego_target_speed,
        }

        # Deterministic blueprint, spawnpoint.
        ego_blueprint = 'vehicle.audi.a2'
        ego_spawn = self.world.map.get_spawn_points()[1]
        ego_location = ego_spawn.location

        self.ego_vehicle = self.world.add_vehicle(SimplexAgent,
                                                  control_params=pid_opt_dict,
                                                  blueprint_filter=ego_blueprint,
                                                  spawn=ego_spawn,
                                                  has_collision_sensor=True,
                                                  has_lane_sensor=True,
                                                  has_dtc_sensor = True,
                                                  ego=True)

    def trajectory_definition(self):
        # Get speed of collision as proportion of target speed.
        ego_collision = [(c[0], c[1] / self.ego_target_speed)
                         for c in self.ego_vehicle.collision_sensor.get_collision_speeds()]
        lane_invasion = self.ego_vehicle.lane_sensor._history.copy()
        # MTL doesn't like empty lists.
        if not ego_collision:
            ego_collision = [(0, -1)]
        if not lane_invasion:
            lane_invasion = [(0, -1)]
        print ('lane_invasion', lane_invasion)
        print ('egocollision', ego_collision)
        traj = {
            'egocollision': ego_collision,
            #'dtc': self.ego_vehicle.control_params['dtc_history'] 
            'laneinvade': lane_invasion 
        }
        return traj

simulation_data = DotMap()
simulation_data.port = PORT
simulation_data.bufsize = BUFSIZE
simulation_data.task = overtake_control_task()

client_task = ClientCarla(simulation_data)
while client_task.run_client():
    pass
print('End of all simulations.')
