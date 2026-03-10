import gymnasium as gym
import highway_env
from stable_baselines3 import PPO

env = gym.make('highway-v0', render_mode='human')


model = PPO(
            "MlpPolicy",
            "highway-fast-v0",
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=32 * 12 // 6,
            batch_size=32,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            tensorboard_log="highway_ppo/",
        )
model.learn(int(2e4))

model.save("highway_ppo/model")

# Load and test saved model
model = PPO.load("highway_ppo/model")
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()