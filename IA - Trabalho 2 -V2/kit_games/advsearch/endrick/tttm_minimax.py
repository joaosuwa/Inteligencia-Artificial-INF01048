from typing import Tuple
from ..tttm.gamestate import GameState
from ..tttm.board import Board
from .minimax import minimax_move


def make_move(state: GameState) -> Tuple[int, int]:
    return minimax_move(state, -1, utility)


def utility(state, player: str) -> float:
    winner = state.winner()
    if winner is None:
        return 0
    return 1 if winner == player else -1
