###
# This code trains a Deep Q-Network (DQN) agent on the "racetrack-v0" environment from the highway-env package.
# Note that DQN requires a discrete action space, which isn't the best suit for the racetrack environment, but
# we are showing an example of how to use DQN with a custom action space. The code also measures the training and testing time.
###

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env
from stable_baselines3 import DQN
import time
import argparse

NUM_STEPS = 20000

def train_dqn():
    start_time = time.time()

    env_train = gym.make("racetrack-v0", config={
        "observation": {
            "type": "Kinematics",  # Kimenatic features for observation to avoid crashing with DQN
            "vehicles_count": 5,
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
            "normalize": True,
        },
        "action": {
            "type": "DiscreteAction"
        }
    })

    model = DQN('MlpPolicy', env_train,
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

    model.learn(NUM_STEPS)
    model.save("racetrack_dqn/model")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed using DQN in {elapsed_time:.2f} seconds.")

    with open("racetrack_dqn/training_time.txt", "w") as f:
        f.write(f"Training completed using DQN in {elapsed_time:.2f} seconds.")


def test_dqn():
    test_start_time = time.time()
    model = DQN.load("racetrack_dqn/model")
    env_test = gym.make("racetrack-v0", render_mode="rgb_array", config={
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
            "normalize": True,
        },
        "action": {
            "type": "DiscreteAction"
        }
    })
    env_test = RecordVideo(env_test, video_folder="racetrack_dqn/videos",
                           episode_trigger=lambda e: True)

    for episode in range(3):
        obs, info = env_test.reset()
        done = truncated = False
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env_test.step(action)

    env_test.close()
    print("Videos saved to racetrack_dqn/videos/")

    test_end_time = time.time()
    test_elapsed_time = test_end_time - test_start_time
    print(f"Testing completed using DQN in {test_elapsed_time:.2f} seconds.")


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

