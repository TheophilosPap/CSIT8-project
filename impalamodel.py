import time
import gymnasium as gym
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecEnvWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from procgen import ProcgenEnv
import numpy as np
from typing import Callable

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


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # The skip connection adds the original input back after two convs.
        # If the convs learn nothing, the block acts as an identity function.
        # This makes it much easier to train deep networks.
        residual = x
        out = nn.functional.relu(x)
        out = self.conv1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        return out + residual  # skip connection

class ImpalaBlock(nn.Module):
    # One "stack": conv -> maxpool -> 2 residual blocks
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaCNN(BaseFeaturesExtractor):
    # Full IMPALA CNN: 3 stacks with increasing channel depth (16, 32, 32)
    # followed by a ReLU, flatten, and a linear layer to produce the feature vector.
    # features_dim=256 matches the output size of the Nature CNN for fair comparison.
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # SB3 passes images as (C, H, W) — channels first
        n_input_channels = observation_space.shape[0]

        self.impala = nn.Sequential(
            ImpalaBlock(n_input_channels, 16),  # stack 1: 16 channels
            ImpalaBlock(16, 32),                # stack 2: 32 channels
            ImpalaBlock(32, 32),                # stack 3: 32 channels
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flat feature size by doing a dummy forward pass
        with th.no_grad():
            sample = th.zeros(1, *observation_space.shape)
            n_flatten = self.impala(sample).shape[1]

        # Final linear layer maps flat features to features_dim
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Normalize pixels from [0, 255] to [0, 1]
        return self.linear(self.impala(observations.float() / 255.0))


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def main():
    print("Initializing 64 parallel environments...")
    venv = ProcgenEnv(num_envs=64, env_name="coinrun", start_level=0, num_levels=0, distribution_mode="hard")
    venv = ProcgenRGBWrapper(venv)
    venv = VecMonitor(venv)

    policy_kwargs = dict(
        features_extractor_class=ImpalaCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    model = PPO(
        policy="CnnPolicy",
        env=venv,
        verbose=1,
        policy_kwargs=policy_kwargs,         
        learning_rate=linear_schedule(5e-4),
        n_steps=256,                         # 256 * 64 = 16,384 samples per update
        batch_size=2048,
        n_epochs=3,
        gamma=0.99,                          # 0.99 confirmed better than 0.999 for CoinRun
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=0.5,
        device="cuda",
        tensorboard_log="./coinrun_tensorboard"
    )

    total_timesteps = 100_000_000
    print(f"Starting IMPALA CNN training for {total_timesteps} steps...")
    start_time = time.time()

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")
    model.save("ppo_coinrun_100mil_impala")
    print("Training complete!")

if __name__ == '__main__':
    main()