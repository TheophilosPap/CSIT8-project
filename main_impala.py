import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecEnvWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from procgen import ProcgenEnv
import numpy as np
from typing import Callable
# https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
class ProcgenRGBWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(15)

    def reset(self):
        obs = self.venv.reset()
        return obs["rgb"]

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs["rgb"], rewards, dones, infos

# impala cnn architecture
# https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        # people call this a resnet style skip connector, input is added to the output of the conv layers
        out = nn.functional.relu(x)
        out = self.conv1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        return out + residual

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv    = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1    = ResidualBlock(out_channels)
        self.res2    = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # SB3 feeds images as (C, H, W) channels first
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            ImpalaBlock(n_input_channels, 16),   
            ImpalaBlock(16, 32),                 
            ImpalaBlock(32, 32),                 
            nn.ReLU(),
            nn.Flatten(),
        )

        # infer the flattened size with a dummy forward pass
        # this is done so that we don't have to manually calculate the size of the flattened output
        # https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html
        with th.no_grad():
            dummy = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # normalize pixels [0, 255] → [0, 1]
        return self.linear(self.cnn(observations.float() / 255.0))

def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func


def main():
    # impala uses a lot of vram, too many env can lead to out of memory error
    env = ProcgenEnv(num_envs=512, env_name="coinrun", start_level=0, num_levels=0, distribution_mode="hard")
    env = ProcgenRGBWrapper(env)
    env = VecMonitor(env)
    print(f"Initializing {env.num_envs} parallel environments...")

    # we pass normalize_images=False so SB3 doesn't divide by 255 before passing to our forward function
    policy_kwargs = dict(
        features_extractor_class=ImpalaCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False, 
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        # if i understood it correctly, it is final + progress * (initial - final)
        learning_rate=linear_schedule(5e-4, 2e-4),
        # number of steps to collect from the environment before updating the model
        n_steps=256,
        # mini batch size, number of samples collected from the environment before updating the model
        batch_size=2048,
        n_epochs=4, 
        # discount factor, probability of the future being relevant
        gamma=0.99, 
        # bias vs variance trade off, lower is more biased but less variance
        gae_lambda=0.95,
        # entropy coefficient, encourages exploration
        ent_coef=0.005,
        # ppo clip range                       
        clip_range=0.2,    
        # value function coefficient, weight of the value function loss
        vf_coef=0.75,
        device="cuda",
        tensorboard_log="./maybeimpala_tensorboard"
    )

    total_timesteps = 100_000_000
    print(f"Starting optimized training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save("ppo_coinrun_100mil_impala")
    print("Training complete!")

if __name__ == '__main__':
    main()
