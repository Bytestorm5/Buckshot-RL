import random
import math
import copy

import abc
from typing import Literal

from buckshot_roulette import BuckshotRoulette
from buckshot_roulette.ai import AbstractEngine


class MCTSNode:
    def __init__(self, state: BuckshotRoulette, parent=None, action=None):
        self.state: BuckshotRoulette = (
            state  # The game state (BuckshotRoulette instance)
        )
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node has been visited
        self.wins = 0  # Total wins (used for backpropagation)
        self.action = action  # The action that led to this node

    def is_fully_expanded(self):
        # A node is fully expanded if all possible moves from this state have been explored
        return len(self.children) == len(self.state.moves())

    def best_child(self, exploration_weight):
        # UCB1 formula: Exploitation + Exploration
        def ucb1(node):
            exploitation = node.wins / node.visits
            exploration = exploration_weight * math.sqrt(
                math.log(self.visits) / node.visits
            )
            return exploitation + exploration

        # Return the child with the highest UCB1 value
        return max(self.children, key=ucb1)

    def expand(self):
        # Expand this node by adding a new child node
        available_moves = self.state.moves()
        explored_moves = [child.action for child in self.children]
        unexplored_moves = [
            move for move in available_moves if move not in explored_moves
        ]

        if unexplored_moves:
            move = random.choice(unexplored_moves)
            next_state = self.state.copy()
            next_state.make_move(move)
            child_node = MCTSNode(next_state, parent=self, action=move)
            self.children.append(child_node)
            return child_node
        return None


class MCTS:
    def __init__(
        self, game: BuckshotRoulette, iterations=1000, exploration_weight=1.414
    ):
        self.game: BuckshotRoulette = game  # The BuckshotRoulette instance
        self.iterations = iterations  # Number of MCTS iterations
        self.exploration_weight = (
            exploration_weight  # Exploration vs Exploitation tradeoff
        )

    def select(self, node: MCTSNode):
        # Selection: Traverse the tree using UCB1 until we reach a leaf node
        while node.children:
            node = node.best_child(self.exploration_weight)
        return node

    def simulate(self, node: MCTSNode):
        # Simulation: Play the game randomly until a terminal state is reached
        current_state = node.state.copy()

        while current_state.winner() is None:
            move = random.choice(current_state.moves())
            current_state.make_move(move)

        return current_state.winner()

    def backpropagate(self, node: MCTSNode, result):
        # Backpropagation: Update the win and visit counts for the node and its ancestors
        while node is not None:
            node.visits += 1
            if result == 0:  # If Player 0 wins
                node.wins += 1
            node = node.parent

    def run(self):
        root = MCTSNode(self.game)  # Root node initialized with the current game state

        for _ in range(self.iterations):
            node = self.select(root)  # Step 1: Selection
            if not node.is_fully_expanded():
                node = node.expand()  # Step 2: Expansion
            result = self.simulate(node)  # Step 3: Simulation
            self.backpropagate(node, result)  # Step 4: Backpropagation

        return root.best_child(0).action  # Return the best action (highest win rate)


class MCTSEngine(AbstractEngine):
    def __init__(
        self,
        playing_as: Literal[0, 1],
        iterations: int = 1000,
        exploration_weight: float = 1.414,
    ):
        super().__init__(playing_as)
        self.iterations = iterations  # Number of iterations for MCTS
        self.exploration_weight = (
            exploration_weight  # Exploration vs exploitation tradeoff
        )

    def choice(self, board: BuckshotRoulette):
        """Determines the best move using MCTS."""
        mcts = MCTS(
            game=board,
            iterations=self.iterations,
            exploration_weight=self.exploration_weight,
        )
        best_move = mcts.run()  # Run MCTS to find the best move
        return best_move

    def post(self, last_move: str, result: any):
        """Post-processing after a move, depending on the result."""
        # Since the game involves magnifying glass and burner phone with hidden information,
        # you could implement logic here to track this.
        # For now, this method doesn't store anything specific but can be customized.
        print(f"Move: {last_move}, Result: {result}")
        # Example logic for handling post-move updates:
        if last_move == "magnifying_glass":
            print(f"Chamber information revealed: {result}")
        elif last_move == "burner_phone":
            print(f"Burner phone result: {result}")


game = BuckshotRoulette(charge_count=3, total_rounds=5, live_rounds=2)
mcts = MCTS(game, iterations=1000)
best_move = mcts.run()
print(f"Best move suggested by MCTS: {best_move}")
