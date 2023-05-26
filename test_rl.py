import sys
import gymnasium
sys.modules["gym"] = gymnasium
import gym_foo
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
print("did imports")


env = gymnasium.make("foo-v0")
print("got env")
model = PPO('CnnPolicy',env,tensorboard_log="RL_logs")
model.learn(20000,progress_bar=True)
model.save("RL_logs/model")