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

# mask template adjusted from https://web.fe.up.pt/~eol/IA/MIA0203/trabalhos/Damas_Othelo/Docs/Eval.html
# could optimize for symmetries but just put all values here for coding speed :P
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

DEPTH = 4


def make_move(state) -> Tuple[int, int]:
    return minimax_move(state, DEPTH, evaluate_mask)


def evaluate_mask(state, player: str) -> float:
    opponent = Board.opponent(player)
    value = 0
    for y in range(8):
        for x in range(8):
            piece = state.board.tiles[y][x]
            if piece == player:
                value += EVAL_TEMPLATE[y][x]
            elif piece == opponent:
                value -= EVAL_TEMPLATE[y][x]
    return float(value)
