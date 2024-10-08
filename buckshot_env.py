import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from copy import deepcopy
from buckshot_roulette import BuckshotRoulette
from buckshot_roulette import Items as BuckshotItems
from buckshot_roulette.ai import Dealer as BuckshotDealer
import torch


class BuckshotRouletteEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(BuckshotRouletteEnv, self).__init__()

        # Initialize the game
        self.game: BuckshotRoulette = BuckshotRoulette()

        # Observation space: Charges, items, current turn, shell knowledge for both players, and new info field
        # The shell knowledge will represent True as 1, False as 0, and None as -1
        self.observation_space = spaces.Dict(
            {
                "charges_self": spaces.Discrete(5),
                "charges_op": spaces.Discrete(5),
                
                "items_player_self_handcuffs": spaces.Discrete(2),
                "items_player_self_magnifying_glass": spaces.Discrete(2),
                "items_player_self_beer": spaces.Discrete(2),
                "items_player_self_saw": spaces.Discrete(2),
                "items_player_self_cigarettes": spaces.Discrete(2),
                "items_player_self_inverter": spaces.Discrete(2),
                "items_player_self_burner_phone": spaces.Discrete(2),
                "items_player_self_meds": spaces.Discrete(2),
                "items_player_self_adrenaline": spaces.Discrete(2),
                
                "items_player_op_handcuffs": spaces.Discrete(2),
                "items_player_op_magnifying_glass": spaces.Discrete(2),
                "items_player_op_beer": spaces.Discrete(2),
                "items_player_op_saw": spaces.Discrete(2),
                "items_player_op_cigarettes": spaces.Discrete(2),
                "items_player_op_inverter": spaces.Discrete(2),
                "items_player_op_burner_phone": spaces.Discrete(2),
                "items_player_op_meds": spaces.Discrete(2),
                "items_player_op_adrenaline": spaces.Discrete(2),
                
                "items_active_handcuffs": spaces.Discrete(2),
                "items_active_magnifying_glass": spaces.Discrete(2),
                "items_active_beer": spaces.Discrete(2),
                "items_active_saw": spaces.Discrete(2),
                "items_active_cigarettes": spaces.Discrete(2),
                "items_active_inverter": spaces.Discrete(2),
                "items_active_burner_phone": spaces.Discrete(2),
                "items_active_meds": spaces.Discrete(2),
                "items_active_adrenaline": spaces.Discrete(2),
                
                "max_charges": spaces.Discrete(3),
                "current_turn": spaces.Discrete(2),
                
                "shell_knowledge_1": spaces.Discrete(3),
                "shell_knowledge_2": spaces.Discrete(3),
                "shell_knowledge_3": spaces.Discrete(3),
                "shell_knowledge_4": spaces.Discrete(3),
                "shell_knowledge_5": spaces.Discrete(3),
                "shell_knowledge_6": spaces.Discrete(3),
                "shell_knowledge_7": spaces.Discrete(3),
                "shell_knowledge_8": spaces.Discrete(3),
                
                "shell_count": spaces.Discrete(9),
                "live_shell_count": spaces.Discrete(9),
                
                "can_handcuffs_0": spaces.Discrete(2),
                "can_magnifying_glass_1": spaces.Discrete(2),
                "can_beer_2": spaces.Discrete(2),
                "can_cigarettes_3": spaces.Discrete(2),
                "can_saw_4": spaces.Discrete(2),
                "can_inverter_5": spaces.Discrete(2),
                "can_burner_phone_6": spaces.Discrete(2),
                "can_meds_7": spaces.Discrete(2),
                "can_adrenaline_8": spaces.Discrete(2),
                "can_op_9": spaces.Discrete(2),
                "can_self_10": spaces.Discrete(2),
            }
        )

        # Action space: The possible moves a player can make
        self.action_space = spaces.Discrete(len(self.game.POSSIBLE_ITEMS) + 2)

        self.POSSIBLE_MOVES = self.game.POSSIBLE_ITEMS.copy()
        self.POSSIBLE_MOVES.append("op")
        self.POSSIBLE_MOVES.append("self")

        # Initialize the shell knowledge for both players
        self.shell_knowledge_player_0 = [None] * 8
        self.shell_knowledge_player_1 = [None] * 8

        self.inverter_uncertainty_0 = False
        self.inverter_uncertainty_1 = False

        self.live_0 = self.game.shotgun_info()[0]
        self.live_1 = self.game.shotgun_info()[0]
        
        self.dealer = BuckshotDealer(1)

    def _get_valid_action_mask(self):
        """
        Create an action mask that invalidates illegal actions.
        """
        possible_moves = self.game.moves()  # Get the list of valid moves
        action_mask = np.zeros(self.action_space.n, dtype=np.float32)

        # Map game moves to action indices
        move_to_index = {item: idx for idx, item in enumerate(self.POSSIBLE_MOVES)}

        # Set the corresponding action index to 1 for valid moves
        for move in possible_moves:
            action_idx = move_to_index.get(move)
            if action_idx is not None:
                action_mask[action_idx] = 1.0

        return action_mask

    def reset(self, seed=None):
        """Resets the environment to an initial state"""
        random.seed(seed)
        self.game = BuckshotRoulette()  # Re-initialize the game

        # Reset the shell knowledge arrays for both players
        self.shell_knowledge_player_0 = [None] * 8
        self.shell_knowledge_player_1 = [None] * 8
        self.inverter_uncertainty_0 = False
        self.inverter_uncertainty_1 = False
        self.live_0 = self.game.shotgun_info()[0]
        self.live_1 = self.game.shotgun_info()[0]

        observation = self._get_observation()
        action_mask = self._get_valid_action_mask()
        
        self.dealer = BuckshotDealer(1)
        
        return observation, action_mask  # Return initial observation

    def step(self, action):
        """Applies an action in the environment"""
        # Get the list of available moves and execute the chosen action
        possible_moves = self.game.moves()
        chosen_move = self.POSSIBLE_MOVES[action]
        current_player = self.game.current_turn

        known_shells = (
            self.shell_knowledge_player_0
            if self.game.current_turn == 0
            else self.shell_knowledge_player_1
        )

        # Perform the chosen move
        result = self.game.make_move(chosen_move)

        # Update the shell knowledge arrays for the current player
        self._update_shell_knowledge(chosen_move, result, known_for=current_player)
        
        while self.game.current_turn != current_player:
            move = self.dealer.choice(self.game)
            res = self.game.make_move(move)
            
            self._update_shell_knowledge(move, res, known_for=1 - current_player)
            self.dealer.post(move, res)
        done = self.game.winner() is not None
        
        known_shells = (
            self.shell_knowledge_player_0
            if self.game.current_turn == 0
            else self.shell_knowledge_player_1
        )
        reward = 1 - (known_shells.count(None) / len(self.game._shotgun))
        if chosen_move in ['op', 'self']:
            reward += result * 10
        # Determine if the game is over and who won
        done = self.game.winner() is not None
        reward = (
            100
            if self.game.winner() == current_player
            else -100 if self.game.winner() == 1 - current_player else 0
        )

        # Get the updated observation
        observation = self._get_observation()

        # Info can include additional information, such as current state
        action_mask = self._get_valid_action_mask()
        info = {
            "result": result,
            "current_turn": self.game.current_turn,
            "action_mask": action_mask,
            "winner": self.game.winner()
        }

        return observation, reward, done, False, info

    def _update_shell_knowledge(self, last_move, move_result, known_for=None):
        """Updates the shell knowledge of the current player based on the move and its result"""
        current_player = self.game.current_turn if known_for == None else known_for
        shell_knowledge = (
            self.shell_knowledge_player_0
            if current_player == 0
            else self.shell_knowledge_player_1
        )

        match last_move:
            case "op", "self", "beer":
                # Remove the first shell as it has been fired
                self.shell_knowledge_player_0.pop(0)
                self.shell_knowledge_player_1.pop(0)

                # Add a new unknown shell
                self.shell_knowledge_player_0.append(None)
                self.shell_knowledge_player_1.append(None)

                self.inverter_uncertainty_0 = False
                self.inverter_uncertainty_1 = False
            case "magnifying_glass":
                shell_knowledge[0] = (
                    1 if move_result else 0
                )  # Update first shell with true (1) or false (0)
                self.inverter_uncertainty_0 = False
                self.inverter_uncertainty_1 = False
            case "burner_phone":
                if move_result != None:
                    shell_knowledge[move_result[0]] = (
                        1 if move_result[1] else 0
                    )  # Update specific shell from burner phone result
            case "inverter":
                if self.shell_knowledge_player_0[0] is None:
                    self.inverter_uncertainty_0 = True
                else:
                    self.shell_knowledge_player_0[0] = (
                        not self.shell_knowledge_player_0[0]
                    )

                if self.shell_knowledge_player_1[0] is None:
                    self.inverter_uncertainty_1 = True
                else:
                    self.shell_knowledge_player_1[0] = (
                        not self.shell_knowledge_player_1[0]
                    )

        if not self.inverter_uncertainty_0:
            self.live_0 = self.game.shotgun_info()[0]
            if self.game._shotgun.count(False) == self.shell_knowledge_player_0.count(
                False
            ):
                self.shell_knowledge_player_0 = [
                    x if x is not None else True for x in self.shell_knowledge_player_0
                ]
            if self.game._shotgun.count(True) == self.shell_knowledge_player_0.count(
                True
            ):
                self.shell_knowledge_player_0 = [
                    x if x is not None else False for x in self.shell_knowledge_player_0
                ]
        if not self.inverter_uncertainty_1:
            self.live_1 = self.game.shotgun_info()[0]
            if self.game._shotgun.count(False) == self.shell_knowledge_player_1.count(
                False
            ):
                self.shell_knowledge_player_1 = [
                    x if x is not None else True for x in self.shell_knowledge_player_1
                ]
            if self.game._shotgun.count(True) == self.shell_knowledge_player_1.count(
                True
            ):
                self.shell_knowledge_player_1 = [
                    x if x is not None else False for x in self.shell_knowledge_player_1
                ]

    def _get_observation(self):
        def item_dict(items: BuckshotItems) -> dict:
            return {
                "handcuffs": int(items.handcuffs > 0),
                "magnifying_glass": int(items.magnifying_glass > 0),
                "beer": int(items.beer > 0),
                "saw": int(items.saw > 0),
                "cigarettes": int(items.cigarettes > 0),
                "inverter": int(items.inverter > 0),
                "burner_phone": int(items.burner_phone > 0),
                "meds": int(items.meds > 0),
                "adrenaline": int(items.adrenaline > 0),
            }
        
        shell_knowledge = (
            self.shell_knowledge_player_0
            if self.game.current_turn == 0
            else self.shell_knowledge_player_1
        )
        
        moves = self.game.moves()
        
        observation = {
            "charges_self": self.game.charges[self.game.current_turn],
            "charges_op": self.game.charges[self.game.opponent()],
            "items_player_self": item_dict(self.game.items[self.game.current_turn]),
            "items_player_op": item_dict(self.game.items[self.game.opponent()]),
            "items_active": item_dict(self.game._active_items),
            "max_charges": self.game.max_charges - 2,  # Because max_charges ranges from 2-4, adjusting to 0-2 for spaces.Discrete(3)
            "current_turn": self.game.current_turn,
            "shell_knowledge": {str(i+1):1 if x is True else 2 if x is False else 0 for i, x in enumerate(shell_knowledge)},
            "shell_count": len(self.game._shotgun),
            "live_shell_count": sum(1 if x else 0 for x in self.game._shotgun),
            "can": {f"{item}_{i}": item in moves for i, item in enumerate(self.POSSIBLE_MOVES)}
        }
        
        out_dict = {}
        for k, v in observation.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    out_dict[f"{k}_{k2}"] = v2
            else:
                out_dict[k] = v

        return out_dict
            

    def render(self, mode="human"):
        """Renders the current state of the game"""
        print(f"Player {self.game.current_turn}'s turn")
        print(f"Charges: {self.game.charges}")
        print(f"Items Player 0: {self.game.items[0]}")
        print(f"Items Player 1: {self.game.items[1]}")
        print(f"Shell Knowledge Player 0: {self.shell_knowledge_player_0}")
        print(f"Shell Knowledge Player 1: {self.shell_knowledge_player_1}")
        print(f"Shell Count: {len(self.game._shotgun)}")
