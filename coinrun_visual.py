import gym
from stable_baselines3 import PPO
import numpy as np
np.bool8 = np.bool

# if i understand it correctly start level is where the clanker starts and num levels is how many levels it goes through
env = gym.make("procgen:procgen-coinrun-v0", start_level=500, num_levels=100, render_mode = "human")

#change this to the model you want to test
model = PPO.load("ppo_coinrun_100mil_impala")

episodes_to_test = 100
episodes_completed = 0
wins = 0

obs = env.reset()
print(f"Starting evaluation over {episodes_to_test} levels...")

while episodes_completed < episodes_to_test:
    # removed deterministic=True, not sure why after the first move it stopped working
    # The agent will now use its stochastic policy to avoid getting stuck
    action, _ = model.predict(obs)
    
    # Step the environment forward
    obs, reward, done, info = env.step(action)
    
    # In CoinRun, the episode ends (done=True) if you get the coin, die, or run out of time.
    if done:
        episodes_completed += 1
        if reward > 0:
            wins += 1
            
        if episodes_completed % 10 == 0:
            print(f"Completed {episodes_completed}/{episodes_to_test} episodes...")
            
        obs = env.reset()

win_probability = (wins / episodes_to_test) * 100

print("\n--- Evaluation Results ---")
print(f"Total Attempts: {episodes_to_test}")
print(f"Total Wins (Coin Collected): {wins}")
print(f"Probability of Winning Per Attempt: {win_probability}%")