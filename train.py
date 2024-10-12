import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from buckshot_env import BuckshotRouletteEnv
import torch
from buckshot_roulette.ai import Dealer
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.dqn.policies import DQNPolicy
from collections import deque
from torch.utils.tensorboard import SummaryWriter

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaskedActorCriticPolicy(MultiInputActorCriticPolicy):
    def forward(self, obs, deterministic=False):
        # Extract the features from the observation
        features = self.extract_features(obs)

        # Separate features for actor and critic if not shared
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Evaluate the value function for the given observations
        values = self.value_net(latent_vf)

        # Get the action distribution from the latent policy features
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Retrieve the valid action mask from obs (assuming it's passed as part of the observation or info)
        if isinstance(obs, dict):
            mask = torch.ones((self.action_space.n, features.shape[0]), device=device)
            for k, v in obs.items():
                comps = k.split("_")
                if comps[0] == "can":
                    mask[int(comps[-1])] = v
        else:
            mask = torch.ones(
                (self.action_space.n, features.shape[0]), device=device
            )  # If no mask is provided, assume all actions are valid

        # Apply the action mask to the action logits
        distribution.distribution.logits = (
            distribution.distribution.logits + (mask.T.float() - 1) * 1e9
        )

        # Sample actions from the distribution (with masked logits)
        actions = distribution.get_actions(deterministic=deterministic)

        # Calculate the log probability of the chosen actions
        log_prob = distribution.log_prob(actions)

        # Reshape the actions to match the action space dimensions
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        return actions, values, log_prob


class MaskedDQNPolicy(DQNPolicy):
    def forward(self, obs):
        # Get the Q-values from the parent class
        q_values = self.q_net(obs)

        # Retrieve the valid action mask from obs or info
        if isinstance(obs, dict) and "valid_action_mask" in obs:
            mask = obs["valid_action_mask"]
        else:
            mask = torch.ones(
                q_values.shape[-1]
            )  # Assume all actions are valid if no mask

        # Set Q-values of invalid actions to a very large negative value
        q_values = q_values + (mask.float() - 1) * 1e9

        return q_values


# Define a custom callback for logging the average reward per game and saving checkpoints
class AverageRewardCheckpointCallback(BaseCallback):
    def __init__(self, checkpoint_freq, verbose=0):
        super(AverageRewardCheckpointCallback, self).__init__(verbose)
        self.checkpoint_freq = checkpoint_freq
        self.curr_game_rewards = 0
        self.win_history = torch.zeros(
            100, device="cuda"
        )  # Store the win history on GPU as a tensor
        self.index = 0  # Circular index to track the position in the win history buffer
        self.total_wins = 0  # Keep track of the total number of wins in the window

    def _on_step(self) -> bool:
        return True

    def _on_step(self) -> None:
        infos = self.locals[
            "infos"
        ]  # This contains information across all environments

        # Win condition tracking
        won_games = [
            info["winner"] == 0
            for info in infos
            if "winner" in info and info["winner"] is not None
        ]

        if len(won_games) > 0:
            won_games_tensor = torch.tensor(
                won_games, dtype=torch.float32, device=device
            )  # Convert to a tensor

            # Update the win history with new values in a circular manner
            num_wins_to_add = min(
                len(won_games_tensor), 100
            )  # Prevent overflow if more than 100 new games
            for win in won_games_tensor[:num_wins_to_add]:
                # Subtract the old value and add the new win to the total
                self.total_wins -= self.win_history[self.index].item()
                self.win_history[self.index] = win
                self.total_wins += win.item()
                # Move to the next index (circular buffer)
                self.index = (self.index + 1) % 100

            # Calculate and log the moving average win rate (SMA100)
            win_rate = self.total_wins / 100.0
            self.logger.record("buckshot/win_rate_sma100", win_rate)

        # Save checkpoint at specified intervals
        if self.n_calls % self.checkpoint_freq == 0 and self.n_calls != 0:
            self.model.save("baseline.zip")

        return True

    def _on_training_end(self) -> None:
        print(f"Training finished at {self.n_steps} steps.")


if __name__ == "__main__":
    # Initialize the environment
    def make_env():
        return BuckshotRouletteEnv()

    # Use SubprocVecEnv for running multiple environments in parallel
    env = DummyVecEnv([lambda: BuckshotRouletteEnv()])

    # Initialize the PPO agent
    if os.path.exists("baseline.zip"):
        agent_player_0 = PPO.load("baseline", env, device=device)
    else:
        agent_player_0 = PPO(
            MaskedActorCriticPolicy,
            env,
            verbose=1,
            tensorboard_log="./log/ad_agent_player",
            device=device,
            n_steps=12500,
        )

    # Define the callback for saving checkpoints
    checkpoint_freq = 100000
    avg_reward_checkpoint_callback = AverageRewardCheckpointCallback(
        checkpoint_freq=checkpoint_freq
    )

    # Train the agent
    n_games = 1000000
    batch_size = 75
    agent_player_0.learn(
        total_timesteps=n_games * batch_size,
        reset_num_timesteps=True,
        progress_bar=True,
        callback=avg_reward_checkpoint_callback,
    )

    # Save the trained model
    agent_player_0.save("baseline")
    print("Training completed and models saved.")
