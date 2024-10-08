import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from buckshot_env import BuckshotRouletteEnv
import torch
from buckshot_roulette.ai import Dealer
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.dqn.policies import DQNPolicy
from collections import deque
from torch.utils.tensorboard import SummaryWriter

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
            mask = torch.ones(self.action_space.n)
            for k, v in obs.items():
                comps = k.split('_')
                if comps[0] == 'can':
                    mask[int(comps[-1])] = v
        else:
            mask = torch.ones(self.action_space.n)  # If no mask is provided, assume all actions are valid

        # Apply the action mask to the action logits
        distribution.distribution.logits = distribution.distribution.logits + (mask.float() - 1) * 1e9

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
        if isinstance(obs, dict) and 'valid_action_mask' in obs:
            mask = obs['valid_action_mask']
        else:
            mask = torch.ones(q_values.shape[-1])  # Assume all actions are valid if no mask

        # Set Q-values of invalid actions to a very large negative value
        q_values = q_values + (mask.float() - 1) * 1e9

        return q_values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the environment
env = BuckshotRouletteEnv()

# Wrap the environment to work with Stable Baselines3
env = DummyVecEnv([lambda: env])  # DummyVecEnv to work with SB3

# Initialize two PPO agents, one for Player 0 and one for Player 1
if os.path.exists("baseline.zip"):
    agent_player_0 = PPO.load("baseline", env, device=device)
else:
    agent_player_0 = PPO(
        MaskedActorCriticPolicy,
        env,
        verbose=1,
        tensorboard_log="./log/ad_agent_player",
        device=device,
    )

n_games = 1000000
train_interval = 2500
batch_size = 75


# Define a custom callback for logging the average reward per game and saving checkpoints
class AverageRewardCheckpointCallback(BaseCallback):
    def __init__(self, checkpoint_freq, verbose=0):
        super(AverageRewardCheckpointCallback, self).__init__(verbose)
        self.checkpoint_freq = checkpoint_freq  # Save checkpoint every 'checkpoint_freq' steps
        self.n_steps = 0
        #self.total_rewards = []
        self.curr_game_rewards = []
        self.win_history = deque(maxlen=100)
        self.last_end = 0

    def _on_step(self) -> bool:
        # Gather rewards from the current episode
        self.n_steps += 1
        rewards = self.locals['rewards']
        self.curr_game_rewards += np.sum(rewards)

        # Log the average reward after each game
        
        if all(self.locals['dones']):  # At the end of a game
            self.logger.record('buckshot/steps_per_game', self.n_steps - self.last_end)
            self.last_end = self.n_steps
            avg_reward = self.curr_game_rewards
            self.curr_game_rewards = 0  # Reset for next game
            self.logger.record('buckshot/reward_per_game', avg_reward)
            
            infos = self.locals['infos']
            won_game = any(info.get('winner', 1) == 0 for info in infos)  # Assuming 'is_success' signals a win
            self.win_history.append(1 if won_game else 0)  # 1 for win, 0 for loss

            # Calculate and log the moving average win rate over the last 100 games
            win_rate = np.mean(self.win_history)
            self.logger.record('buckshot/win_rate_sma100', win_rate)

        # Save checkpoint at specified intervals
        if self.n_steps % self.checkpoint_freq == 0:
            checkpoint_path = f"ppo_checkpoint_{self.n_steps}_steps.zip"
            self.model.save(checkpoint_path)
            self.model.save('baseline.zip')
            print(f"Checkpoint saved at {self.n_steps} steps to {checkpoint_path}")

        return True

    def _on_training_end(self) -> None:
        # Any final operations at the end of training can be done here
        print(f"Training finished at {self.n_steps} steps.")


# Initialize the callback with a checkpoint frequency, e.g., save every 100,000 steps
checkpoint_freq = 100000
avg_reward_checkpoint_callback = AverageRewardCheckpointCallback(checkpoint_freq=checkpoint_freq)

# Train the agent with the callback
agent_player_0.learn(
    total_timesteps=n_games * batch_size,
    reset_num_timesteps=True,
    progress_bar=True,
    callback=avg_reward_checkpoint_callback  # Add the custom callback here
)

# Save the trained agents
agent_player_0.save("baseline")

print("Training completed and models saved.")

