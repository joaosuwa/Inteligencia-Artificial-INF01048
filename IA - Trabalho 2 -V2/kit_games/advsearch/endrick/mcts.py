import math
import random
from typing import Tuple

from ..othello.board import Board


class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = list(state.legal_moves())
        random.shuffle(self.untried_moves)

    def ucb1(self, exploration=1.41):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self):
        return max(self.children, key=lambda c: c.ucb1())

    def expand(self):
        move = self.untried_moves.pop()
        next_state = self.state.next_state(move)
        child = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child)
        return child

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        return self.state.is_terminal()


def rollout(state):
    current = state
    while not current.is_terminal():
        moves = list(current.legal_moves())
        if not moves:
            break
        move = random.choice(moves)
        current = current.next_state(move)
    winner = current.winner()
    if winner is None:
        return 0
    return 1 if winner == state.player else -1


def backpropagate(node, result, original_player):
    while node is not None:
        node.visits += 1
        if node.state.player != original_player:
            node.wins -= result
        else:
            node.wins += result
        node = node.parent


def make_move(state) -> Tuple[int, int]:
    iterations = 1000
    root = MCTSNode(state)

    for _ in range(iterations):
        node = root

        while not node.is_terminal() and node.is_fully_expanded() and node.children:
            node = node.best_child()

        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        result = rollout(node.state)
        backpropagate(node, result, state.player)

    return root.best_child().move
