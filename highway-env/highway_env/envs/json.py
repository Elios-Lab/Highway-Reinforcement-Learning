import numpy as np
import openpyxl
from openpyxl import Workbook
import copy
import json
import jsonschema
from jsonschema import validate
import os
from typing import List, Tuple, Union, Optional, Callable
import gym
import random
from gym import Wrapper
from gym.utils import seeding
from gym.envs.registration import register
from highway_env import utils

LaneIndex = Tuple[str, str, int]
Action = Union[int, np.ndarray]

# Describe what kind of json you expect.
paramsRLSchema = {
    "type": "object",
    "properties": {
        "lanes_count": {
            "type": "number"
            },
        "vehicles_count": {
            "type": "number"
            },
        "initial_lane_id": {
            "type": "number"
            },
        "duration": {
            "type": "number"
            },
        "ego_spacing": {
            "type": "number"
            },
        "vehicles_density": {
            "type": "number"
            },
        "collision_reward": {
            "type": "number"
            },
        "limit_speed": {
            "type": "number"
            },        
        "initial_speed_ego": {
            "type": "number"
            },
        "High_Speed_Reward": {
            "type": "number"
            },
        "Speed_Reward": {
            "type": "number"
            },
        "Right_Overtaking_Reward": {
            "type": "number"
            },
        "Unsafe_Distance_Reward": {
            "type": "number"
            },
        "Right_Lane_Reward": {
            "type": "number"
            },
        "Left_Overtaking_Reward": {
            "type": "number"
            },
        "Long_Accel_Reward": {
            "type": "number"
            },
        "Long_Decel_Reward": {
            "type": "number"
            },
        "Hazardous_Lane_Change_Reward": {
            "type": "number"
            },
        "Steering_Reward": {
            "type": "number"
        },
        "SafeDistance": {
            "type": "number"
            } 
    },
    "required": ["lanes_count", "vehicles_count", "initial_lane_id", "duration", "ego_spacing", "vehicles_density", "collision_reward", "limit_speed", "initial_speed_ego", "High_Speed_Reward", "Speed_Reward", "Right_Overtaking_Reward", "Unsafe_Distance_Reward", "Right_Lane_Reward", "Left_Overtaking_Reward", "Long_Accel_Reward", "Long_Decel_Reward", "Hazardous_Lane_Change_Reward", "Steering_Reward", "SafeDistance"],
    "additionalProperties": False
}

# Describe what kind of json you expect.
paramsSchema = {
    "type": "object",
    "properties": {
        "lanes_count": {
            "type": "number"
            },
        "vehicles_count": {
            "type": "number"
            },
        "initial_lane_id": {
            "type": "number"
            },
        "duration": {
            "type": "number"
            },
        "ego_spacing": {
            "type": "number"
            },
        "vehicles_density": {
            "type": "number"
            },       
        "initial_speed_ego": {
            "type": "number"
            },
        "SafeDistance": {
            "type": "number"
            } 
    },
    "required": ["lanes_count", "vehicles_count", "initial_lane_id", "duration", "ego_spacing", "vehicles_density", "initial_speed_ego", "SafeDistance"],
    "additionalProperties": False
}

# Describe what kind of json you expect.
paramsWeightSchema = {
    "type": "object",
    "properties": {
        "weight_crush": {
            "type": "number"
            },
        "weight_km": {
            "type": "number"
            },
        "weight_right_overtake": {
            "type": "number"
            },
        "weight_left_overtake": {
            "type": "number"
            },
        "weight_AVG_TTC": {
            "type": "number"
            },
        "weight_STD_TTC": {
            "type": "number"
            },
        "weight_AVG_DSL": {
            "type": "number"
            },
        "weight_STD_DSL": {
            "type": "number"
            },        
        "weight_AVG_LOAF": {
            "type": "number"
            },
        "weight_number_lane_change": {
            "type": "number"
            },
        "weight_hazardous": {
            "type": "number"
            },
        "weight_AVG_ACC": {
            "type": "number"
            },
        "weight_STD_ACC": {
            "type": "number"
            },
        "weight_AVG_DEC": {
            "type": "number"
            },
        "weight_STD_DEC": {
            "type": "number"
            },
        "weight_AVG_STER": {
            "type": "number"
            },
        "weight_STD_STER": {
            "type": "number"
            }
    },
    "required": ["weight_crush", "weight_km", "weight_right_overtake", "weight_left_overtake", "weight_AVG_TTC", "weight_STD_TTC", "weight_AVG_DSL", "weight_STD_DSL", "weight_AVG_LOAF", "weight_number_lane_change", "weight_hazardous", "weight_AVG_ACC", "weight_STD_ACC", "weight_AVG_DEC", "weight_STD_DEC", "weight_AVG_STER", "weight_STD_STER"],
    "additionalProperties": False
}

# Describe what kind of json you expect.
paramsEvaluationFunctionSchema = {
    "type": "object",
    "properties": {
        "max_crush_per_h": {
            "type": "number"
            },
        "max_km_per_h": {
            "type": "number"
            },
        "max_right_overtake_per_h": {
            "type": "number"
            },
        "max_left_overtake_per_h": {
            "type": "number"
            },
        "max_AVG_TTC_per_h": {
            "type": "number"
            },
        "max_STD_TTC_per_h": {
            "type": "number"
            },
        "max_AVG_DSL_per_h": {
            "type": "number"
            },
        "max_STD_DSL_per_h": {
            "type": "number"
            },        
        "max_AVG_LOAF_per_h": {
            "type": "number"
            },
        "max_number_lane_change_per_h": {
            "type": "number"
            },
        "max_hazardous_per_h": {
            "type": "number"
            },
        "max_AVG_ACC_per_h": {
            "type": "number"
            },
        "max_STD_ACC_per_h": {
            "type": "number"
            },
        "max_AVG_DEC_per_h": {
            "type": "number"
            },
        "max_STD_DEC_per_h": {
            "type": "number"
            },
        "max_AVG_STER_per_h": {
            "type": "number"
            },
        "max_STD_STER_per_h": {
            "type": "number"
            }
    },
    "required": ["max_crush_per_h", "max_km_per_h", "max_right_overtake_per_h", "max_left_overtake_per_h", "max_AVG_TTC_per_h", "max_STD_TTC_per_h", "max_AVG_DSL_per_h", "max_STD_DSL_per_h", "max_AVG_LOAF_per_h", "max_number_lane_change_per_h", "max_hazardous_per_h", "max_AVG_ACC_per_h", "max_STD_ACC_per_h", "max_AVG_DEC_per_h", "max_STD_DEC_per_h", "max_AVG_STER_per_h", "max_STD_STER_per_h"],
    "additionalProperties": False
}

paramsEvaluationEpSchema = {
    "type": "object",
    "properties": {
        "max_crush": {
            "type": "number"
            },
        "max_km": {
            "type": "number"
            },
        "max_right_overtake": {
            "type": "number"
            },
        "max_left_overtake": {
            "type": "number"
            },
        "max_AVG_TTC": {
            "type": "number"
            },
        "max_STD_TTC": {
            "type": "number"
            },
        "max_AVG_DSL": {
            "type": "number"
            },
        "max_STD_DSL": {
            "type": "number"
            },        
        "max_AVG_LOAF": {
            "type": "number"
            },
        "max_number_lane_change": {
            "type": "number"
            },
        "max_hazardous": {
            "type": "number"
            },
        "max_AVG_ACC": {
            "type": "number"
            },
        "max_STD_ACC": {
            "type": "number"
            },
        "max_AVG_DEC": {
            "type": "number"
            },
        "max_STD_DEC": {
            "type": "number"
            },
        "max_AVG_STER": {
            "type": "number"
            },
        "max_STD_STER": {
            "type": "number"
            }
    },
    "required": ["max_crush", "max_km", "max_right_overtake", "max_left_overtake", "max_AVG_TTC", "max_STD_TTC", "max_AVG_DSL", "max_STD_DSL", "max_AVG_LOAF", "max_number_lane_change", "max_hazardous", "max_AVG_ACC", "max_STD_ACC", "max_AVG_DEC", "max_STD_DEC", "max_AVG_STER", "max_STD_STER"],
    "additionalProperties": False
}

class Json(gym.Env):

    def config_json_rl (self):
        with open(os.environ.get('HIGHWAY_ENV_PATH') + 'paramsRL.json') as config_file:
            data = json.load(config_file)
        try:
            validate(instance= data, schema=paramsRLSchema)
            if data['lanes_count'] > 15 or data['lanes_count'] <= 0 or not isinstance(data['lanes_count'], int):
                data['lanes_count'] = 3
            if data['vehicles_count'] <= 0 or not isinstance(data['initial_lane_id'], int):
                data['vehicles_count'] = 10
            if data['initial_lane_id'] >= data['lanes_count'] or data['initial_lane_id'] < 0 or not isinstance(data['initial_lane_id'], int):
                data['initial_lane_id'] = data['lanes_count'] - 1
            if data['duration'] <= 0 or not isinstance(data['duration'], int):
                data['duration'] = 120      
            if data['ego_spacing'] > 1 or data['ego_spacing'] <= 0 or not isinstance(data['ego_spacing'], float):
                data['ego_spacing'] = 0.5
            if data['vehicles_density'] > 1 or data['vehicles_density'] <= 0 or not isinstance(data['vehicles_density'], float):
                data['vehicles_density'] = 0.7
            if data['initial_speed_ego'] > 50 and data['initial_speed_ego'] <= 0 or not isinstance(data['initial_speed_ego'], int):
                data['initial_speed_ego'] = 25
            if data['collision_reward'] > 0 or data['collision_reward'] < -1 or not isinstance(data['collision_reward'], int):
                data['collision_reward'] = -1  
            if data['limit_speed'] > 50 or data['limit_speed'] <= 5 or not isinstance(data['limit_speed'], int):
                data['limit_speed'] = 30
            if data['Right_Lane_Reward'] > 0 or data['Right_Lane_Reward'] < -1:
                data['Right_Lane_Reward'] = -0.3
            if data['High_Speed_Reward'] < 0 or data['High_Speed_Reward'] > 1:
                data['High_Speed_Reward'] = 0.4
            if data['Speed_Reward'] > 0 or data['Speed_Reward'] < -1:
                data['Speed_Reward'] = -0.4
            if data['Right_Overtaking_Reward'] > 0 or data['Right_Overtaking_Reward'] < -1:
                data['Right_Overtaking_Reward'] = -0.75
            if data['Unsafe_Distance_Reward'] > 0 or data['Unsafe_Distance_Reward'] < -1:
                data['Unsafe_Distance_Reward'] = -0.3
            if data['Left_Overtaking_Reward'] < 0 or data['Left_Overtaking_Reward'] > 1:
                data['Left_Overtaking_Reward'] = 0.75
            if data['Long_Accel_Reward'] > 0 or data['Long_Accel_Reward'] < -1:
                data['Long_Accel_Reward'] = -0.45
            if data['Long_Decel_Reward'] > 0 or data['Long_Decel_Reward'] < -1:
                data['Long_Decel_Reward'] = -0.35
            if data['Hazardous_Lane_Change_Reward'] > 0 or data['Hazardous_Lane_Change_Reward'] < -1:
                data['Hazardous_Lane_Change_Reward'] = -0.3
            if data['Steering_Reward'] > 0 or data['Steering_Reward'] < -1:
                data['Steering_Reward'] = -0.1
            if not isinstance(data['SafeDistance'], int):
                data['SafeDistance'] = 1
        except:
            data['lanes_count'] = 3
            data['vehicles_count'] = 10
            data['initial_lane_id'] = data['lanes_count'] - 1
            data['duration'] = 120      
            data['ego_spacing'] = 0.5
            data['vehicles_density'] = 0.7
            data['collision_reward'] = -1            
            data['limit_speed'] = 30
            data['initial_speed_ego'] = 25
            data['High_Speed_Reward'] = 0.4
            data['Speed_Reward'] = -0.4
            data['Right_Overtaking_Reward'] = -0.75
            data['Unsafe_Distance_Reward'] = -0.5
            data['Right_Lane_Reward'] = -0.5
            data['Left_Overtaking_Reward'] = 0.75
            data['Long_Accel_Reward'] = -0.1
            data['Long_Decel_Reward'] = -0.1
            data['Hazardous_Lane_Change_Reward'] = -0.1
            data["Steering_Reward"] = -0.1
            data['SafeDistance'] = 1 
            print("ERROR! LOADED DEFAULT PARAMS!")         
        return(data) 

        
    
    def config_json (self):
        with open(os.environ.get('HIGHWAY_ENV_PATH') + 'params.json') as config_file:
            data = json.load(config_file)
        try:
            validate(instance= data, schema=paramsSchema)
            if data['lanes_count'] > 15 or data['lanes_count'] <= 0 or not isinstance(data['lanes_count'], int):
                data['lanes_count'] = 3
            if data['vehicles_count'] <= 0 or not isinstance(data['initial_lane_id'], int):
                data['vehicles_count'] = 10
            if data['initial_lane_id'] >= data['lanes_count'] or data['initial_lane_id'] < 0 or not isinstance(data['initial_lane_id'], int):
                data['initial_lane_id'] = data['lanes_count'] - 1
            if data['duration'] <= 0 or not isinstance(data['duration'], int):
                data['duration'] = 120      
            if data['ego_spacing'] > 1 or data['ego_spacing'] <= 0 or not isinstance(data['ego_spacing'], float):
                data['ego_spacing'] = 0.5
            if data['vehicles_density'] > 1 or data['vehicles_density'] <= 0 or not isinstance(data['vehicles_density'], float):
                data['vehicles_density'] = 0.7
            if data['initial_speed_ego'] > 50 and data['initial_speed_ego'] <= 0 or not isinstance(data['initial_speed_ego'], int):
                data['initial_speed_ego'] = 25
            if not isinstance(data['SafeDistance'], int):
                data['SafeDistance'] = 1
        except:
            data['lanes_count'] = 3
            data['vehicles_count'] = 10
            data['initial_lane_id'] = data['lanes_count'] - 1
            data['duration'] = 120      
            data['ego_spacing'] = 0.5
            data['vehicles_density'] = 0.7
            data['initial_speed_ego'] = 25            
            data['SafeDistance'] = 1 
            print("ERROR! LOADED DEFAULT PARAMS!")
        return(data)           

    def config_json_weight (self):
        with open(os.environ.get('HIGHWAY_ENV_PATH') + 'weightparamsEvaluationFunction.json') as config_file:
            data = json.load(config_file)
        try:
            validate(instance= data, schema=paramsWeightSchema)
            if not isinstance(data['weight_crush'], int):
                data['weight_crush'] = 15
            if not isinstance(data['weight_km'], int):
                data['weight_km'] = 15
            if not isinstance(data['weight_right_overtake'], int):
                data['weight_right_overtake'] = 11
            if not isinstance(data['weight_left_overtake'], int):
                data['weight_left_overtake'] = 12      
            if not isinstance(data['weight_AVG_TTC'], int):
                data['weight_AVG_TTC'] = 5
            if not isinstance(data['weight_STD_TTC'], int):
                data['weight_STD_TTC'] = 2
            if not isinstance(data['weight_AVG_DSL'], int):
                data['weight_AVG_DSL'] = 5
            if not isinstance(data['weight_STD_DSL'], int):
                data['weight_STD_DSL'] = 2  
            if not isinstance(data['weight_AVG_LOAF'], int):
                data['weight_AVG_LOAF'] = 8
            if not isinstance(data['weight_number_lane_change'], int):
                data['weight_number_lane_change'] = 5
            if not isinstance(data['weight_hazardous'], int):
                data['weight_hazardous'] = 8
            if not isinstance(data['weight_AVG_ACC'], int):
                data['weight_AVG_ACC'] = 2
            if not isinstance(data['weight_STD_ACC'], int):
                data['weight_STD_ACC'] = 2
            if not isinstance(data['weight_AVG_DEC'], int):
                data['weight_AVG_DEC'] = 2
            if not isinstance(data['weight_STD_DEC'], int):
                data['weight_STD_DEC'] = 2
            if not isinstance(data['weight_AVG_STER'], int):
                data['weight_AVG_STER'] = 2
            if not isinstance(data['weight_STD_STER'], int):
                data['weight_STD_STER'] = 2
        except:
            data['weight_crush'] = 15
            data['weight_km'] = 15
            data['weight_right_overtake'] = 11
            data['weight_left_overtake'] = 12      
            data['weight_AVG_TTC'] = 5
            data['weight_STD_TTC'] = 2
            data['weight_AVG_DSL'] = 5
            data['weight_STD_DSL'] = 2            
            data['weight_AVG_LOAF'] = 8
            data['weight_number_lane_change'] = 5
            data['weight_hazardous'] = 8
            data['weight_AVG_ACC'] = 2
            data['weight_STD_ACC'] = 2
            data['weight_AVG_DEC'] = 2
            data['weight_STD_DEC'] = 2
            data['weight_AVG_STER'] = 2
            data['weight_STD_STER'] = 2
            print("ERROR! LOADED DEFAULT WEIGHT!")
        return(data)         

    def config_json_max (self):
        with open(os.environ.get('HIGHWAY_ENV_PATH') + 'paramsEvaluationFunction.json') as config_file:
            data = json.load(config_file)
        try:
            validate(instance= data, schema=paramsEvaluationFunctionSchema)
            if not isinstance(data['max_crush_per_h'], int):
                data['max_crush_per_h'] = 4
            if not isinstance(data['max_km_per_h'], int):
                data['max_km_per_h'] = 100
            if not isinstance(data['max_right_overtake_per_h'], int):
                data['max_right_overtake_per_h'] = 12
            if not isinstance(data['max_left_overtake_per_h'], int):
                data['max_left_overtake_per_h'] = 120      
            if not isinstance(data['max_AVG_TTC_per_h'], int):
                data['max_AVG_TTC_per_h'] = 8
            if not isinstance(data['max_STD_TTC_per_h'], int):
                data['max_STD_TTC_per_h'] = 1
            if not isinstance(data['max_AVG_DSL_per_h'], int):
                data['max_AVG_DSL_per_h'] = 5
            if not isinstance(data['max_STD_DSL_per_h'], int):
                data['max_STD_DSL_per_h'] = 1 
            if not isinstance(data['max_AVG_LOAF_per_h'], int):
                data['max_AVG_LOAF_per_h'] = 2
            if not isinstance(data['max_number_lane_change_per_h'], int):
                data['max_number_lane_change_per_h'] = 400
            if not isinstance(data['max_hazardous_per_h'], int):
                data['max_hazardous_per_h'] = 10
            if not isinstance(data['max_AVG_ACC_per_h'], float):
                data['max_AVG_ACC_per_h'] = 0.5
            if not isinstance(data['max_STD_ACC_per_h'], int):
                data['max_STD_ACC_per_h'] = 1
            if not isinstance(data['max_AVG_DEC_per_h'], float):
                data['max_AVG_DEC_per_h'] = 0.1
            if not isinstance(data['max_STD_DEC_per_h'], int):
                data['max_STD_DEC_per_h'] = 1
            if not isinstance(data['max_AVG_STER_per_h'], float):
                data['max_AVG_STER_per_h'] = 0.1
            if not isinstance(data['max_STD_STER_per_h'], int):
                data['max_STD_STER_per_h'] = 1
        except:
            data['max_crush_per_h'] = 4
            data['max_km_per_h'] = 100
            data['max_right_overtake_per_h'] = 12
            data['max_left_overtake_per_h'] = 120     
            data['max_AVG_TTC_per_h'] = 8
            data['max_STD_TTC_per_h'] = 1
            data['max_AVG_DSL_per_h'] = 5
            data['max_STD_DSL_per_h'] = 1           
            data['max_AVG_LOAF_per_h'] = 2
            data['max_number_lane_change_per_h'] = 400
            data['max_hazardous_per_h'] = 10
            data['max_AVG_ACC_per_h'] = 0.5
            data['max_STD_ACC_per_h'] = 1
            data['max_AVG_DEC_per_h'] = 0.1
            data['max_STD_DEC_per_h'] = 1
            data['max_AVG_STER_per_h'] = 0.1
            data['max_STD_STER_per_h'] = 1
            print("ERROR! LOADED DEFAULT WEIGHT!")
        return(data) 

    def config_json_max_ep (self):
        with open(os.environ.get('HIGHWAY_ENV_PATH') + 'paramsEpisode.json') as config_file:
            data = json.load(config_file)
        try:
            validate(instance= data, schema=paramsEvaluationEpSchema)
            if not isinstance(data['max_crush'], int):
                data['max_crush'] = 0
            if not isinstance(data['max_km'], float):
                data['max_km'] = 1.8
            if not isinstance(data['max_right_overtake'], int):
                data['max_right_overtake'] = 1
            if not isinstance(data['max_left_overtake'], int):
                data['max_left_overtake'] = 5      
            if not isinstance(data['max_AVG_TTC'], int):
                data['max_AVG_TTC'] = 8
            if not isinstance(data['max_STD_TTC'], int):
                data['max_STD_TTC'] = 1
            if not isinstance(data['max_AVG_DSL'], int):
                data['max_AVG_DSL'] = 5
            if not isinstance(data['max_STD_DSL'], int):
                data['max_STD_DSL'] = 1  
            if not isinstance(data['max_AVG_LOAF'], int):
                data['max_AVG_LOAF'] = 2
            if not isinstance(data['max_number_lane_change'], int):
                data['max_number_lane_change'] = 10
            if not isinstance(data['max_hazardous'], int):
                data['max_hazardous'] = 1
            if not isinstance(data['max_AVG_ACC'], float):
                data['max_AVG_ACC'] = 0.5
            if not isinstance(data['max_STD_ACC'], int):
                data['max_STD_ACC'] = 1
            if not isinstance(data['max_AVG_DEC'], float):
                data['max_AVG_DEC'] = 0.1
            if not isinstance(data['max_STD_DEC'], int):
                data['max_STD_DEC'] = 1
            if not isinstance(data['max_AVG_STER'], float):
                data['max_AVG_STER'] = 0.1
            if not isinstance(data['max_STD_STER'], int):
                data['max_STD_STER'] = 1
        except:
            data['max_crush'] = 0
            data['max_km'] = 1.8
            data['max_right_overtake'] = 1
            data['max_left_overtake'] = 5     
            data['max_AVG_TTC'] = 8
            data['max_STD_TTC'] = 1
            data['max_AVG_DSL'] = 5
            data['max_STD_DSL'] = 1           
            data['max_AVG_LOAF'] = 2
            data['max_number_lane_change'] = 10
            data['max_hazardous'] = 1
            data['max_AVG_ACC'] = 0.5
            data['max_STD_ACC'] = 1
            data['max_AVG_DEC'] = 0.1
            data['max_STD_DEC'] = 1
            data['max_AVG_STER'] = 0.1
            data['max_STD_STER'] = 1
            print("ERROR! LOADED DEFAULT WEIGHT!")
        return(data)     