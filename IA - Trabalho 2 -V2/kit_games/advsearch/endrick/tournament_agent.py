from typing import Tuple
from ..othello.gamestate import GameState
from .minimax import minimax_move_id
from .othello_minimax_custom import evaluate_custom


def make_move(state: GameState) -> Tuple[int, int]:
    return minimax_move_id(state, 30, evaluate_custom)
