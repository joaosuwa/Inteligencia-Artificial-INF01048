from typing import Tuple
from ..othello.gamestate import GameState
from ..othello.board import Board
from .minimax import minimax_move_id


def make_move(state) -> Tuple[int, int]:
    return minimax_move_id(state, 4, evaluate_count)


def evaluate_count(state: GameState, player: str) -> float:
    board = state.board
    opponent = Board.opponent(player)
    return board.num_pieces(player) - board.num_pieces(opponent)
