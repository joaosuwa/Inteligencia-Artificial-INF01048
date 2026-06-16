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

DEPTH = 4


def make_move(state) -> Tuple[int, int]:
    return minimax_move(state, DEPTH, evaluate_count)


def evaluate_count(state, player: str) -> float:
    opponent = Board.opponent(player)
    player_pieces = state.board.num_pieces(player)
    opponent_pieces = state.board.num_pieces(opponent)
    return float(player_pieces - opponent_pieces)
