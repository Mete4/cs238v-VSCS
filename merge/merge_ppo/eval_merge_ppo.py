import numpy as np
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from tqdm import trange

gym.register_envs(highway_env)

model = PPO.load("PPO_1/final_model.zip")
env = gym.make("merge-v0")

NUM_EPISODES = 100000
successes = 0
failures = 0

collision_positions = []
collision_speeds = []
collision_types = []

for episode in trange(NUM_EPISODES):
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
    else:
        successes += 1

env.close()

# Save results so you can plot without re-running
np.save("collision_positions.npy", np.array(collision_positions))
np.save("collision_speeds.npy", np.array(collision_speeds))
np.save("collision_types.npy", np.array(collision_types))

merger_fails  = collision_types.count('merger')
highway_fails = collision_types.count('highway')
p_fail = failures / NUM_EPISODES

print(f"Successes:      {successes}/{NUM_EPISODES}")
print(f"Failures:       {failures}/{NUM_EPISODES}")
print(f"  Merger:       {merger_fails} ({merger_fails/NUM_EPISODES*100:.2f}%)")
print(f"  Highway:      {highway_fails} ({highway_fails/NUM_EPISODES*100:.2f}%)")
print(f"P(failure) ≈    {p_fail:.4f}  ({p_fail*100:.2f}%)")
