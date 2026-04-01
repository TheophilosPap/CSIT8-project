import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecEnvWrapper
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

class RewardShapingWrapper(VecEnvWrapper):
    def __init__(self, venv, time_penalty=0.005):
        super().__init__(venv)

        self.time_penalty = time_penalty

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        shaped_rewards = rewards.copy()

        shaped_rewards -= self.time_penalty

        return obs, shaped_rewards, dones, infos


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def main():
    print("Initializing 256 parallel environments...")
    env = ProcgenEnv(num_envs=256, env_name="coinrun", start_level=0, num_levels=10000, distribution_mode="hard")
    env = ProcgenRGBWrapper(env)
    env = RewardShapingWrapper(env, time_penalty=0.005)
    env = VecMonitor(env)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(5e-4), # higher start, shrinking to zero
        n_steps=256,                         # 256 * 256 = 65,536 samples per update
        batch_size=2048,
        n_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        clip_range=0.2,
        vf_coef=0.5,
        device="cuda",
        tensorboard_log="./coinrun_tensorboard"
    )

    total_timesteps = 100_000_000
    print(f"Starting optimized training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save("coinrun_100mil_shaped")
    print("Training complete!")

if __name__ == '__main__':
    main()