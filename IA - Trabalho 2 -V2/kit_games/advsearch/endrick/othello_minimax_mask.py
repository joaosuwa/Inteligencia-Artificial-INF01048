from typing import Tuple
from ..othello.gamestate import GameState
from ..othello.board import Board
from .minimax import minimax_move_id

# DO NOT CHANGE!
EVAL_TEMPLATE = [
    [100, -30, 6, 2, 2, 6, -30, 100],
    [-30, -50, 1, 1, 1, 1, -50, -30],
    [  6,   1, 1, 1, 1, 1,   1,   6],
    [  2,   1, 1, 3, 3, 1,   1,   2],
    [  2,   1, 1, 3, 3, 1,   1,   2],
    [  6,   1, 1, 1, 1, 1,   1,   6],
    [-30, -50, 1, 1, 1, 1, -50, -30],
    [100, -30, 6, 2, 2, 6, -30, 100]
]


def make_move(state) -> Tuple[int, int]:
    return minimax_move_id(state, 4, evaluate_mask)


def evaluate_mask(state: GameState, player: str) -> float:
    board = state.board
    opponent = Board.opponent(player)
    player_sum = 0
    opponent_sum = 0
    for y in range(8):
        for x in range(8):
            piece = board.tiles[y][x]
            if piece == player:
                player_sum += EVAL_TEMPLATE[y][x]
            elif piece == opponent:
                opponent_sum += EVAL_TEMPLATE[y][x]
    return player_sum - opponent_sum
