import argparse
import os
from datetime import datetime
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.dqn import DQN
from stable_baselines3.common.logger import configure
from minigrid.wrappers import ImgObsWrapper



class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

if __name__ == "__main__":

    #  Parser ----------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    #  Directory Paths
    parser.add_argument('-model', '--model_dir', default="Models", help="Agent's model directory", type=str, required=False)
    parser.add_argument('-log', '--log_dir', default="Logs", help="Agent's log directory", type=str, required=False)
    #  Policy name
    parser.add_argument('-policy', '--policy_name', default="test", help="Filename of the agent's policy", type=str, required=True)
    #  Hyper-parameters
        # DQN
    parser.add_argument('-limit', '--timestep_limit', default=100000, help="Limits for training", type=int,
                        required=True)
    parser.add_argument('-batch', '--batch_size', default=32, help="Batch size", type=int, required=False)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, help="Learning rate", type=float, required=False)
    parser.add_argument('-df', '--discount_factor', default=.99, help="Discount factor", type=float, required=False)

    parser.add_argument('-exp_fraction', '--exploration_fraction', default=0.1, help="Number of steps where epsilon decrease", type=float, required=False)
    parser.add_argument('-exp_s', '--exploration_initial_eps', default=1.0, help="Epsilon' starting value", type=float, required=False)
    parser.add_argument('-exp_f', '--exploration_final_eps', default=0.05, help="Epsilon' final value", type=float, required=False)

    parser.add_argument('-sync', '--target_update_interval', default=10000, help="Synchronize target net at each n steps", type=int, required=False)
    parser.add_argument('-replay', '--buffer_size', default=1000000, help="Size of replay memory", type=int, required=False)
    parser.add_argument('-replay_s', '--learning_starts', default=50000, help="From which number of experiences NN training process start", type=int, required=False)

        # Env
    parser.add_argument('-map', '--map_name', default="MiniGrid-Dynamic-Obstacles-5x5-v0", help="Map's name", type=str, required=False)

    args = parser.parse_args()

    # Get arguments
    LOG_DIR = args.log_dir
    MODEL_DIR = args.model_dir

    FILENAME = args.policy_name

    TIME_LIMIT = args.timestep_limit
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    DISCOUNT_FACTOR = args.discount_factor

    EXP_FRACTION = args.exploration_fraction
    EXP_START = args.exploration_initial_eps
    EXP_FINAL = args.exploration_final_eps

    TARGET_UPDATE = args.target_update_interval
    BUFFER_SIZE = args.buffer_size
    LEARNING_START = args.learning_starts

    MAP = args.map_name

    # ------------------------------------------------------------------------------------------------------------------

    #  Create Dirs to store logs and models
    if not os.path.exists(LOG_DIR):  # log dir
        os.mkdir(LOG_DIR)
    if not os.path.exists(MODEL_DIR):  # model dir
        os.mkdir(MODEL_DIR)

    # Shape agent's states for DQN
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    # Env initialisation
    env = gym.make(MAP) # , render_mode="human"
    env = ImgObsWrapper(env)

    # Set up logger
    new_logger = configure(LOG_DIR, ["stdout", "csv"])

    # Model initialisation
    model = DQN("CnnPolicy", env, learning_rate=LEARNING_RATE, buffer_size=BUFFER_SIZE,
                learning_starts=LEARNING_START, batch_size=BATCH_SIZE, gamma=DISCOUNT_FACTOR,
                target_update_interval=TARGET_UPDATE, exploration_fraction=EXP_FRACTION,
                exploration_initial_eps=EXP_START, exploration_final_eps=EXP_FINAL,
                policy_kwargs=policy_kwargs, verbose=1)

    # Training
    model.set_logger(new_logger)
    model.learn(TIME_LIMIT)

    # Store
    now = datetime.now().strftime("%a_%b_%m__%I:%M:%S__%p_%Y")
    model.save(MODEL_DIR + os.sep + FILENAME + '-' + now)

    del model

