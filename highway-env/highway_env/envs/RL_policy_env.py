import numpy as np
from typing import Tuple
from gym.envs.registration import register

import json
import jsonschema
from jsonschema import validate
import os

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork, LaneIndex
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.envs.statistic import Statistics
from highway_env.envs.json import Json

LaneIndex = Tuple[str, str, int]

class RLPolicyEnv(AbstractEnv):

    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """Desired maximum deceleration."""
    
    #lists used to collect data
    list_vel = [] #machine speed list at each step
    list_over = [] #list that collects 0 if you are not passing a machine on the right in that step and 5 if you are passing it
    list_unsafe = [] #list that collects 0 if you are at a safe distance and 5 if you are not respecting the distance
    list_hlc = [] #list that collect the number of Hazardous Lane Change that ego_vehicle makes respect its new rear_vehicle
    list_right = [] #list that collects 0 if you are not in the lane further to the right and 1 if you are
    list_nChangeLane = [] #list that collects the lane in which you are at each step
    list_evaluate = [] #list that collects, in order according to the columns of the excel file, all the statistics of an episode
    list_leftID = [] # list that collects all the IDs of the cars passed on the left
    list_rightID = [] # list that collects all the IDs of the cars passed on the right
    list_acc = [] #list of acceleration
    list_dec = [] #list of deceleration
    list_str = [] #list of steering
    list_l = []
    list_stdttc = [] #list of ttc values for each steps     
    list_stddsl = [] #list of vel values for each steps
    list_stdacc = [] #list of acc values for each steps
    list_stddec = [] #list of dec values for each steps
    list_stdstr = [] #list of steering values for each steps
    list_idhaz = []
    list_action = [] 
    list_hour = []
    list_perc = []
    list_p = []
    lista = [] 
    start = 0.0 

    def default_config(self) -> dict:
        config = super().default_config()
        data = Json.config_json_rl(self)
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction", #type of actions (see, common / action line 143)
            },
            "lanes_count": data['lanes_count'],
            "vehicles_count": data['vehicles_count'],
            "initial_lane_id": data['initial_lane_id'],
            "duration": data['duration'], 
            "ego_spacing": data['ego_spacing'], 
            "vehicles_density": data['vehicles_density'],
            "initial_speed_ego": data['initial_speed_ego'],
            "collision_reward": data['collision_reward'],  #the reward received when colliding with a vehicle
            "limit_speed": data['limit_speed'],
            "Right_Lane_Reward" : data['Right_Lane_Reward'], #reward for traveling in the right lane
            "High_Speed_Reward" : data['High_Speed_Reward'], #reward for traveling at a speed 
            "Speed_Reward" : data['Speed_Reward'], #reward for traveling at a speed 
            "Right_Overtaking_Reward" : data['Right_Overtaking_Reward'], #negative reward for overtaking on the right
            "Unsafe_Distance_Reward" : data['Unsafe_Distance_Reward'], #negative reward for not keeping a safe distance
            "Left_Overtaking_Reward" : data['Left_Overtaking_Reward'],
            "Long_Accel_Reward" : data['Long_Accel_Reward'],
            "Long_Decel_Reward" : data['Long_Decel_Reward'],
            "Hazardous_Lane_Change_Reward" : data['Hazardous_Lane_Change_Reward'],
            "Steering_Reward" : data['Steering_Reward'],
            "SafeDistance" : data['SafeDistance'], #which was choosen algorith for safedistance reward 
            "Lane_Hazardous_Reward" : -0.5,
            "controlled_vehicles": 1,
            "env" : 0, #parameter added to understand which env is running, see abstract line 201
            "offroad_terminal": False
        })
        return config
    
    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        i = 1
        self.controlled_vehicles = []
        for _ in range(self.config["controlled_vehicles"]):
            vehicle = self.action_type.vehicle_class.create_random(i,
                                                                   self.road,
                                                                   speed=self.config["initial_speed_ego"],
                                                                   lane_id=self.config["initial_lane_id"],
                                                                   spacing=self.config["ego_spacing"])
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle) 
            i += 1

            vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
            for _ in range(self.config["vehicles_count"]):
                    self.road.vehicles.append(vehicles_type.create_random(i, self.road, spacing=1 / self.config["vehicles_density"]))
                    i += 1
        
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, to avoid collisions and to not overtake on the right.
        :param action: the last action performed
        :return: the corresponding reward
        """ 
               
        
        #richiama il metodo liste in Statistics che ritorna se sono state superate macchine sulla dx, se non viene mantenuta la distanza di sicurezza
        #e se si viaggia sulla corsia più a destra
        overtakeCount, unsafe, vel, leftOvertake, acc, dec, hazardousLC, steering, vel_pos, lane = Statistics.liste(self, ego_vehicle = self.vehicle, action = action, distance_algo = self.config["SafeDistance"]) 

        lane_haz = 0
        if len(self.list_action) < 2:
            self.list_action.append(action)
            if len(self.list_action) == 2:
                if self.list_action == [0,2] or self.list_action == [2,0]:
                    lane_haz = 1
                del self.list_action[0]
        

        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["Right_Overtaking_Reward"] * overtakeCount\
            + self.config["Unsafe_Distance_Reward"] * unsafe \
            + self.config["Right_Lane_Reward"] * lane \
            + self.config["Speed_Reward"] * vel \
            + self.config["High_Speed_Reward"] * vel_pos \
            + self.config["Long_Accel_Reward"] * (acc/2 \
            + self.config["Long_Decel_Reward"] * (dec/2) \
            + self.config["Left_Overtaking_Reward"] * leftOvertake \
            + self.config["Hazardous_Lane_Change_Reward"] * hazardousLC \
            + self.config["Lane_Hazardous_Reward"] * lane_haz \
            + self.config["Steering_Reward"] * steering  

        reward = utils.lmap(reward,
                          [self.config["collision_reward"]+ self.config["Lane_Hazardous_Reward"]+ self.config["Right_Overtaking_Reward"]+ self.config["Unsafe_Distance_Reward"]+ self.config["Right_Lane_Reward"] + self.config["Speed_Reward"] + self.config["Long_Accel_Reward"]+ self.config["Long_Decel_Reward"]+ self.config["Hazardous_Lane_Change_Reward"]+ self.config["Steering_Reward"], self.config["Left_Overtaking_Reward"]+ self.config["High_Speed_Reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward

        

        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] 
            

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)
    
register(
    id='rl-policy-v0',
    entry_point='highway_env.envs:RLPolicyEnv',
)
