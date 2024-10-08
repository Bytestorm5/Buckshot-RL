import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from copy import deepcopy
from buckshot_roulette import BuckshotRoulette
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
                "items_player_self": spaces.Dict(
                    {
                        "handcuffs": spaces.Discrete(2),
                        "magnifying_glass": spaces.Discrete(2),
                        "beer": spaces.Discrete(2),
                        "saw": spaces.Discrete(2),
                        "cigarettes": spaces.Discrete(2),
                        "inverter": spaces.Discrete(2),
                        "burner_phone": spaces.Discrete(2),
                        "meds": spaces.Discrete(2),
                        "adrenaline": spaces.Discrete(2),
                    }
                ),
                "items_player_op": spaces.Dict(
                    {
                        "handcuffs": spaces.Discrete(2),
                        "magnifying_glass": spaces.Discrete(2),
                        "beer": spaces.Discrete(2),
                        "saw": spaces.Discrete(2),
                        "cigarettes": spaces.Discrete(2),
                        "inverter": spaces.Discrete(2),
                        "burner_phone": spaces.Discrete(2),
                        "meds": spaces.Discrete(2),
                        "adrenaline": spaces.Discrete(2),
                    }
                ),
                "items_active": spaces.Dict(
                    {
                        "handcuffs": spaces.Discrete(2),
                        "magnifying_glass": spaces.Discrete(2),
                        "beer": spaces.Discrete(2),
                        "saw": spaces.Discrete(2),
                        "cigarettes": spaces.Discrete(2),
                        "inverter": spaces.Discrete(2),
                        "burner_phone": spaces.Discrete(2),
                        "meds": spaces.Discrete(2),
                        "adrenaline": spaces.Discrete(2),
                    }
                ),
                "max_charges": spaces.Discrete(3),
                "current_turn": spaces.Discrete(2),
                "shell_knowledge": spaces.Dict(
                    {
                        "1": spaces.Discrete(3, start=-1),
                        "2": spaces.Discrete(3, start=-1),
                        "3": spaces.Discrete(3, start=-1),
                        "4": spaces.Discrete(3, start=-1),
                        "5": spaces.Discrete(3, start=-1),
                        "6": spaces.Discrete(3, start=-1),
                        "7": spaces.Discrete(3, start=-1),
                        "8": spaces.Discrete(3, start=-1),
                    }
                ),
                "shell_count": spaces.Discrete(9),
                "live_shell_count": spaces.Discrete(9),
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
        return observation, action_mask  # Return initial observation

    def step(self, action):
        """Applies an action in the environment"""
        # Get the list of available moves and execute the chosen action
        # possible_moves = self.game.moves()
        chosen_move = self.POSSIBLE_MOVES[action]
        current_player = self.game.current_turn

        known_shells = (
            self.shell_knowledge_player_0
            if self.game.current_turn == 0
            else self.shell_knowledge_player_1
        )
        known_heur = 1 - (known_shells.count(None) / len(self.game._shotgun))

        # Perform the chosen move
        result = self.game.make_move(chosen_move)

        # Update the shell knowledge arrays for the current player
        self._update_shell_knowledge(chosen_move, result, known_for=current_player)

        known_shells = (
            self.shell_knowledge_player_0
            if self.game.current_turn == 0
            else self.shell_knowledge_player_1
        )
        known_heur = (
            1 - (known_shells.count(None) / len(self.game._shotgun))
        ) - known_heur

        # Determine if the game is over and who won
        done = self.game.winner() is not None
        reward = (
            1
            if self.game.winner() == current_player
            else -1 if self.game.winner() == 1 - current_player else 0
        )

        # Get the updated observation
        observation = self._get_observation()

        # Info can include additional information, such as current state
        action_mask = self._get_valid_action_mask()
        info = {
            "result": result,
            "current_turn": self.game.current_turn,
            "action_mask": action_mask,
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
        """Helper function to gather the current game state"""
        shell_knowledge = (
            self.shell_knowledge_player_0
            if self.game.current_turn == 0
            else self.shell_knowledge_player_1
        )
        return {
            "charges_self": self.game.charges[self.game.current_turn],
            "charges_op": self.game.charges[1 - self.game.current_turn],
            "items_player_self": np.array(
                [
                    self.game.items[self.game.current_turn][item]
                    for item in self.game.POSSIBLE_ITEMS
                ],
                dtype=np.int32,
            ),
            "items_player_op": np.array(
                [
                    self.game.items[1 - self.game.current_turn][item]
                    for item in self.game.POSSIBLE_ITEMS
                ],
                dtype=np.int32,
            ),
            "current_turn": self.game.current_turn,
            "shell_knowledge": np.array(
                [1 if x is True else -1 if x is False else 0 for x in shell_knowledge],
                dtype=np.int32,
            ),
            "shell_count": len(self.game._shotgun),
            "live_shell_count": (
                self.live_0 if self.game.current_turn == 0 else self.live_1
            ),
        }

    def render(self, mode="human"):
        """Renders the current state of the game"""
        print(f"Player {self.game.current_turn}'s turn")
        print(f"Charges: {self.game.charges}")
        print(f"Items Player 0: {self.game.items[0]}")
        print(f"Items Player 1: {self.game.items[1]}")
        print(f"Shell Knowledge Player 0: {self.shell_knowledge_player_0}")
        print(f"Shell Knowledge Player 1: {self.shell_knowledge_player_1}")
        print(f"Shell Count: {len(self.game._shotgun)}")
