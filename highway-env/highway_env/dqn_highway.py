import gym
import highway_env
import numpy as np
import pprint

from stable_baselines import DQN

env = gym.make("highway-v0")

env.config["lanes_count"] = 4
env.config["vehicles_count"] = 50
env.config["duration"] = 40
env.config["other_vehicles_type"] = "highway_env.vehicle.behavior.IDMVehicle"
env.config["centering_position"] = [0.3, 0.5]

env.config["screen_height"] = 300
env.config["screen_width"] = 1200
env.config["scaling"] = 10

# Load saved model (trained in Google Colab)
model = DQN.load('dqn_highway_15e4', env=env)

# Print environment configuration
pprint.pprint(env.config)

obs = env.reset()
for _ in range(80):
    env.render()
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        env.reset()
env.close()