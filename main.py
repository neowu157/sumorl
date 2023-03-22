import sys
import sumo_rl
import os
import traci

from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from sumo_rl import SumoEnvironment


env = SumoEnvironment(
    net_file='environment.net.xml',
    route_file='episode_routes.rou.xml',
    out_csv_name="outputs/2way-single-intersection/dqn",
    single_agent=True,
    use_gui=False,
    num_seconds=100000,
)


save_path = os.path.join('Training', 'Saved Models')
log_path = os.path.join('Training', 'Logs')


model = DQN(
    env=env,
    policy="MlpPolicy",
    learning_rate=0.001,
    learning_starts=0,
    train_freq=1,
    target_update_interval=500,
    exploration_initial_eps=0.05,
    exploration_final_eps=0.01,
    verbose=1,
    tensorboard_log=log_path
)

model.learn(total_timesteps=10)
model.save(save_path)
print('finish');
