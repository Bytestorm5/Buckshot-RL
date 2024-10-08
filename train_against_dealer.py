import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from buckshot_env import BuckshotRouletteEnv
import torch
from buckshot_roulette.ai import Dealer
import os
from stable_baselines3.common.callbacks import BaseCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the environment
env = BuckshotRouletteEnv()

# Wrap the environment to work with Stable Baselines3
env = DummyVecEnv([lambda: env])  # DummyVecEnv to work with SB3

# Initialize two PPO agents, one for Player 0 and one for Player 1
if os.path.exists('baseline.zip'):
    agent_player_0 = PPO.load("baseline", env, device=device)
else:
    agent_player_0 = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log='./ad_agent_player', device=device)
agent_player_dealer = Dealer(1)

def preprocess_observation(observation):
    """
    Convert the observation from an OrderedDict to a format that can be processed by the policy.
    """
    # Flatten or convert the observation dictionary to a format that the agent can use
    return {key: torch.tensor(value).float() for key, value in observation.items()}

def mask_action_probabilities(agent, observation, action_mask):
    """
    Mask the invalid actions in the action probability distribution and renormalize.
    """
    # Preprocess the observation
    observation = preprocess_observation(observation)

    # Ensure the observation is on the correct device (CPU/GPU)
    observation_tensor = {k: v.to(agent.device) for k, v in observation.items()}

    # Get the action distribution from the policy
    action_distribution = agent.policy.get_distribution(observation_tensor)

    # Extract the action probabilities
    action_probs = action_distribution.distribution.probs

    # Apply the action mask (ensure it's on the correct device)
    action_mask_tensor = torch.as_tensor(action_mask, dtype=torch.float32).to(agent.device)

    # Mask invalid actions by multiplying the action probabilities with the mask
    masked_action_probs = action_probs * action_mask_tensor

    # Check if any valid actions remain
    if masked_action_probs.sum() == 0:
        raise ValueError("No valid actions to choose from after masking.")

    # Normalize the masked probabilities (prevent division by zero)
    masked_action_probs /= masked_action_probs.sum()

    # Sample an action from the masked probabilities
    action = torch.multinomial(masked_action_probs, 1).item()

    return action

n_games = 1000000  # Define how many games to play
train_interval = 2500  # Train after every 10 games
batch_size = 75  # Number of timesteps per game

current_WR = 0.0
needs_update = False

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        global needs_update, current_WR
        # Log scalar value (here a random variable)
        if needs_update:
            needs_update = False
            self.logger.record("train/p0_win_rate", current_WR)
        return True

agent_player_0.learn(total_timesteps=n_games / train_interval, reset_num_timesteps=False, progress_bar=True, callback=TensorboardCallback())

# Save the trained agents
agent_player_0.save("baseline")

print("Training completed and models saved.")
