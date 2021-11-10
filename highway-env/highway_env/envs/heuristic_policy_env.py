import numpy as np
import openpyxl
from typing import Tuple
from openpyxl import Workbook

import json
import jsonschema
from jsonschema import validate
import os

from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action, DiscreteMetaAction
from highway_env.road.road import Road, RoadNetwork, LaneIndex
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.envs.statistic import Statistics
from highway_env.envs.json import Json

class HeuristicPolicy(AbstractEnv):
    
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
    list_rightID = []
    list_acc = [] #list of acceleration
    list_dec = [] #list of deceleration
    list_str = [] #list of steering
    list_stdttc = [] #list of ttc values for each steps     
    list_stddsl = [] #list of vel values for each steps
    list_stdacc = [] #list of acc values for each steps
    list_stddec = [] #list of dec values for each steps
    list_stdstr = [] #list of steering values for each steps
    list_idhaz = []
    list_hour = []
    list_l = []
    list_perc = []
    list_p = []
    lista = [] 
    start = 0.0 


    def default_config(self) -> dict:
        config = super().default_config()
        data = Json.config_json(self)
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": data['lanes_count'],
            "vehicles_count": data['vehicles_count'],
            "initial_lane_id": data['initial_lane_id'],
            "duration": data['duration'], 
            "ego_spacing": data['ego_spacing'], 
            "vehicles_density": data['vehicles_density'],
            "initial_speed_ego": data['initial_speed_ego'],
            "SafeDistance" : data['SafeDistance'], 
            "limit_speed": 30,
            "number_episodes" : 3,            
            "controlled_vehicles": 1,
            "env" : 2,
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
    
    #we have left this method but it is irrelevant with the policy since it always returns 0. 
    #It is needed because it is called up at each step for the statistics
    def _reward(self, action: Action) -> float:
                    
        #richiama il metodo liste in Statistics che ritorna se sono state superate macchine sulla dx, se non viene mantenuta la distanza di sicurezza
        #e se si viaggia sulla corsia più a destra
        overtakeCount, unsafe, vel, leftOvertake, acc, dec, hazardousLC, steering, vel_pos, lane = Statistics.liste(self, ego_vehicle = self.vehicle, action = action, distance_algo = self.config["SafeDistance"]) 
        reward=0
        return reward

    #method which the colab in the last cell is called and which returns the action to be performed
    def action_heuristic(self):
        actions = AbstractEnv.get_available_actions(self)
        heuristic_action = IDMVehicle.get_heuristic_action(self, ego_vehicle = self.vehicle, lane_index = self.vehicle.lane_index, actions = actions)
        return heuristic_action

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] 
            

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)        
    
register(
    id='heuristic-policy-v0',
    entry_point='highway_env.envs:HeuristicPolicy',
)
