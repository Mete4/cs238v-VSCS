import numpy as np
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from tqdm import trange

gym.register_envs(highway_env)

model = PPO.load("merge/merge_ppo/PPO_1/final_model")
env = gym.make("merge-v0")

NUM_EPISODES = 100000
CHECKPOINT_INTERVAL = 1000   # record p_fail estimate every this many episodes

successes = 0
failures = 0

collision_positions = []
collision_speeds = []
collision_types = []
collision_episode_indices = []   # which episode each collision occurred in

# Running p_fail estimate sampled every CHECKPOINT_INTERVAL episodes
checkpoint_episodes = []   # episode number at each checkpoint
checkpoint_p_fail   = []   # cumulative p_fail estimate at each checkpoint

pbar = trange(NUM_EPISODES)
for episode in pbar:
    obs, info = env.reset()
    done = truncated = False
    merger = env.unwrapped.road.vehicles[-1]

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

    vehicle = env.unwrapped.vehicle
    if bool(getattr(vehicle, 'crashed', False)):
        failures += 1
        collision_positions.append(vehicle.position.copy())
        collision_speeds.append(vehicle.speed)
        hit_merger = bool(merger.crashed)
        collision_types.append('merger' if hit_merger else 'highway')
        collision_episode_indices.append(episode)
    else:
        successes += 1

    # Record checkpoint and update tqdm postfix
    episodes_done = episode + 1
    if episodes_done % CHECKPOINT_INTERVAL == 0:
        p_fail_now = failures / episodes_done
        checkpoint_episodes.append(episodes_done)
        checkpoint_p_fail.append(p_fail_now)
        pbar.set_postfix(p_fail=f"{p_fail_now:.5f}", failures=failures)

env.close()

# --- Save results ---
np.save("collision_positions.npy",        np.array(collision_positions))
np.save("collision_speeds.npy",           np.array(collision_speeds))
np.save("collision_types.npy",            np.array(collision_types, dtype=object))
np.save("collision_episode_indices.npy",  np.array(collision_episode_indices))
np.save("checkpoint_episodes.npy",        np.array(checkpoint_episodes))
np.save("checkpoint_p_fail.npy",          np.array(checkpoint_p_fail))

merger_fails  = collision_types.count('merger')
highway_fails = collision_types.count('highway')
p_fail = failures / NUM_EPISODES

print(f"\nSuccesses:      {successes}/{NUM_EPISODES}")
print(f"Failures:       {failures}/{NUM_EPISODES}")
print(f"  Merger:       {merger_fails} ({merger_fails/NUM_EPISODES*100:.2f}%)")
print(f"  Highway:      {highway_fails} ({highway_fails/NUM_EPISODES*100:.2f}%)")
print(f"P(failure) ≈    {p_fail:.5f}  ({p_fail*100:.3f}%)")

# Normal approximation 95% CI for final estimate
import math
margin = 1.96 * math.sqrt(p_fail * (1 - p_fail) / NUM_EPISODES)
print(f"95% CI:         [{p_fail - margin:.5f}, {p_fail + margin:.5f}]")
