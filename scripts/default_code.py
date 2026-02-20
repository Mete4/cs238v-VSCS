import gymnasium as gym
import highway_env

env = gym.make('highway-v0', render_mode='human')
# env = gym.make('racetrack-v0', render_mode='human')

obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action = env.action_space.sample()  # Take random action
    obs, reward, done, truncated, info = env.step(action)