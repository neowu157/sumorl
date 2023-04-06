import sys
import os
import gymnasium as gym
import numpy as np
from stable_baselines3.dqn.dqn import DQN
from sumo_rl import SumoEnvironment
import datetime
from visualization import Visualization
from utility import import_train_configuration, set_sumo, set_train_path


class CustomWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_neg_reward = 0
        self.sum_queue_length = 0
        self.sum_waiting_time = 0
        self.sum_speed = 0

    def reset(self, **kwargs):
        self.total_neg_reward = 0
        self.sum_queue_length = 0
        self.sum_waiting_time = 0
        self.sum_speed = 0
        return self.env.reset(**kwargs)

    def step(self, action):

        observation, rewards, terminated, truncated, info = self.env.step(action)

        if rewards < 0:
            self.total_neg_reward += (rewards*100)    #plotreward

        self.sum_queue_length += info['system_total_stopped'] #vehicle speed is below 0.1 m/s
        self.sum_waiting_time += info['system_total_stopped'] # for every stopped car +1s per 1 step  plotdelay

        vehicles = self.sumo.vehicle.getIDList()
        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        self.sum_speed  += np.sum(speeds)


        print(info['step'],rewards)
        if info['step']== 7200:
            print(self.total_neg_reward)           #plotreward  update every 5 step
            print(self.sum_waiting_time)    #plotdelay
            print(self.sum_queue_length/7200)  #plotqueue
            print(self.sum_speed/7200)   #plotspeed

        return observation, rewards, terminated, truncated, info

timestamp_start = datetime.datetime.now()

path = set_train_path('models')

Visualization = Visualization(
    path,
    dpi=96
)

env = gym.make('sumo-rl-v0',
    net_file='environment.net.xml',
    route_file='episode_routes.rou.xml',
    out_csv_name="outputs/dqn",
    single_agent=True,
    use_gui=True,
    num_seconds=7200,
)
env = CustomWrapper(env)


#save_path = os.path.join('Saved Models')


model = DQN(
    env=env,
    policy="MlpPolicy",
    learning_rate=0.001,
    learning_starts=0,
    train_freq=1,
    target_update_interval=500,
    exploration_initial_eps=0.05,
    exploration_final_eps=0.01,
    verbose=1
)

print(model.policy)

model.learn(total_timesteps=1500,progress_bar=True)
#model.save(path)

print("\n----- Start time:", timestamp_start)
print("----- End time:", datetime.datetime.now())
print("----- Session info saved at:", path)
