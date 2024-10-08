import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from buckshot_env import BuckshotRouletteEnv

# Load the trained bot model for Player 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent_player_1 = PPO.load("agent_player_1", device=device)

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

# Initialize the environment
env = BuckshotRouletteEnv()
env = DummyVecEnv([lambda: env])
observation = env.reset()  # Reset the environment for a new game
action_mask = np.zeros(11)
action_mask[-1] = 1.0
action_mask[-2] = 1.0
done = False

print("Welcome to Human vs Bot! You are Player 0 (Human).")

total_reward_player_0 = 0
total_reward_player_1 = 0

while not done:
    current_player = env.envs[0].game.current_turn
    
    if current_player == 0:
        # Human's turn
        env.envs[0].render()
        
        print("Your turn, Player 0!")
        #action_mask = observation['action_mask']  # Use the action mask from the environment
        valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1.0]
        
        print(f"Valid actions: {valid_actions}")        
        for action_idx in valid_actions:
            print(f" - {action_idx}: {env.envs[0].POSSIBLE_MOVES[action_idx]}")
        
        action = int(input(f"Choose your action: "))
        
        if action not in valid_actions:
            print("Invalid action! Please try again.")
            continue  # Skip the step to re-prompt for valid input
        
        #observation, reward, done, _, info = env.step(action)    
        
    else:
        # Bot's turn (Player 1)
        print("Bot's turn (Player 1).")
        #action_mask = observation
        action = mask_action_probabilities(agent_player_1, observation, action_mask)
        print(f"Bot chooses action {action}")
    print()
    # Execute the action in the environment
    observation, reward, done, info = env.step([action])
    reward = reward[0]
    done = done[0]
    action_mask = info[0]['action_mask']
    # Accumulate rewards for each player
    if current_player == 0:
        total_reward_player_0 += reward
    else:
        total_reward_player_1 += reward
    
    #action_mask = info['action_mask']
    
    if done:
        print(f"Game finished! Player 0 (You) Reward: {total_reward_player_0}, Player 1 (Bot) Reward: {total_reward_player_1}")
        break

print("Thank you for playing!")
