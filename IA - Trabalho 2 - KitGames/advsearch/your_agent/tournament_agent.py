import random
from typing import Tuple
from ..othello.gamestate import GameState
from ..othello.board import Board
from .minimax import minimax_move
from .othello_minimax_custom import evaluate_custom

# Voce pode criar funcoes auxiliares neste arquivo
# e tambem modulos auxiliares neste pacote.
#
# Nao esqueca de renomear 'your_agent' com o nome
# do seu agente.

DEPTH = 5


def make_move(state) -> Tuple[int, int]:
    if state.game_name == 'Othello':
        return minimax_move(state, DEPTH, evaluate_custom)


