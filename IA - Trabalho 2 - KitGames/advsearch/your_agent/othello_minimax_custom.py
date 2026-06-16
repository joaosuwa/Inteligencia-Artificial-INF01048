import random
from typing import Tuple
from ..othello.gamestate import GameState
from ..othello.board import Board
from .minimax import minimax_move

# Voce pode criar funcoes auxiliares neste arquivo
# e tambem modulos auxiliares neste pacote.
#
# Nao esqueca de renomear 'your_agent' com o nome
# do seu agente.

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

DEPTH = 4


def make_move(state) -> Tuple[int, int]:
    return minimax_move(state, DEPTH, evaluate_custom)


def evaluate_custom(state, player: str) -> float:
    if state.is_terminal():
        winner = state.winner()
        if winner == player:
            return 1000000.0
        elif winner is None:
            return 0.0
        else:
            return -1000000.0

    opponent = Board.opponent(player)
    empty_count = state.board.num_pieces(Board.EMPTY)
    total = 0.0

    pos_val = 0
    for y in range(8):
        for x in range(8):
            piece = state.board.tiles[y][x]
            if piece == player:
                pos_val += EVAL_TEMPLATE[y][x]
            elif piece == opponent:
                pos_val -= EVAL_TEMPLATE[y][x]
    total += pos_val

    player_moves = len(state.board.legal_moves(player))
    opp_moves = len(state.board.legal_moves(opponent))
    total += 12 * (player_moves - opp_moves)

    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    for x, y in corners:
        piece = state.board.tiles[y][x]
        if piece == player:
            total += 25
        elif piece == opponent:
            total -= 25

    if empty_count <= 16:
        total += state.board.num_pieces(player) - state.board.num_pieces(opponent)

    return total
