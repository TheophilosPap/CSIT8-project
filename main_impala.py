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


# ── Environment wrapper (same as currentmain_modelcreator) ─────────────────────
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


# ── IMPALA CNN (SB3 equivalent of build_impala_cnn) ────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = nn.functional.relu(x)
        out = self.conv1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        return out + residual


class ImpalaBlock(nn.Module):
    """One IMPALA stack: conv → maxpool → 2 residual blocks."""
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
    """
    SB3 features extractor that replicates build_impala_cnn from baselines.
    Three stacks (16 → 32 → 32 channels), followed by ReLU + flatten + linear.

    Usage:
        policy_kwargs = dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs)
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # SB3 feeds images as (C, H, W) — channels first
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            ImpalaBlock(n_input_channels, 16),   # stack 1
            ImpalaBlock(16, 32),                 # stack 2
            ImpalaBlock(32, 32),                 # stack 3
            nn.ReLU(),
            nn.Flatten(),
        )

        # Infer the flattened size with a dummy forward pass
        with th.no_grad():
            dummy = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Normalize pixels [0, 255] → [0, 1]
        return self.linear(self.cnn(observations.float() / 255.0))


# ── Helpers ────────────────────────────────────────────────────────────────────
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Initializing 256 parallel environments...")
    env = ProcgenEnv(num_envs=256, env_name="coinrun", start_level=0, num_levels=5000, distribution_mode="hard")
    env = ProcgenRGBWrapper(env)
    env = VecMonitor(env)

    # This is the SB3 equivalent of:
    #   from baselines.common.models import build_impala_cnn
    #   model = build_impala_cnn(...)
    policy_kwargs = dict(
        features_extractor_class=ImpalaCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=5e-4,
        n_steps=256,
        batch_size=2048, # do not make it too big, it can take too much vram and crash
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        clip_range=0.2,
        vf_coef=0.5,
        device="cuda",
        tensorboard_log="./comp_coinrun_tensorboard"
    )

    total_timesteps = 50_000_000
    print(f"Starting IMPALA CNN training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save("coinrun_50mil_impala_hard")
    print("Training complete!")


if __name__ == '__main__':
    main()
