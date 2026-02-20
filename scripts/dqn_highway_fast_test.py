###
# This code trains a Deep Q-Network (DQN) agent on "highway-v0"
###

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env
from stable_baselines3 import DQN
import time
import argparse

def train_dqn():
    start_time = time.time()

    env = gym.make("highway-fast-v0")
    model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log="highway_dqn/")
    model.learn(int(2e4))
    model.save("highway_dqn/model")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed using DQN in {elapsed_time:.2f} seconds.")

    with open("highway_dqn/training_time.txt", "w") as f:
        f.write(f"Training completed using DQN in {elapsed_time:.2f} seconds.")


def test_dqn():
    model = DQN.load("highway_dqn/model")
    env_test = gym.make("highway-fast-v0", render_mode="rgb_array")
    env_test = RecordVideo(env_test, video_folder="highway_dqn/videos",
                           episode_trigger=lambda e: True)

    for episode in range(3):
        obs, info = env_test.reset()
        done = truncated = False
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env_test.step(action)

    env_test.close()
    print("Videos saved to racetrack_dqn/videos/")


def main():
    # Add test flag
    parser = argparse.ArgumentParser(description="Train and test DQN on racetrack environment.")
    parser.add_argument("--test", action="store_true", help="Only run the testing phase.")
    args = parser.parse_args()
    if args.test:
        test_dqn()
    else:
        train_dqn()
        test_dqn()
    # env = gym.make("racetrack-v0")
    # print(env.unwrapped.config)


if __name__ == "__main__":
    main()

