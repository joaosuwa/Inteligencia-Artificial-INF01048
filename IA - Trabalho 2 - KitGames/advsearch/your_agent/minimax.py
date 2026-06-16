import random
from typing import Tuple, Callable


def minimax_move(state, max_depth: int, eval_func: Callable) -> Tuple[int, int]:
    def alphabeta(state, depth_left, alpha, beta, maximizing, eval_func, root_player):
        if state.is_terminal():
            return eval_func(state, root_player)

        if depth_left == 0:
            return eval_func(state, root_player)

        moves = state.legal_moves()
        if not moves:
            opponent = 'W' if state.player == 'B' else 'B'
            board_copy = state.board.copy()
            pass_state = state.__class__(board_copy, opponent)
            return alphabeta(pass_state, depth_left, alpha, beta, not maximizing, eval_func, root_player)

        if maximizing:
            value = -float('inf')
            for move in moves:
                next_state = state.next_state(move)
                value = max(value, alphabeta(next_state, depth_left - 1, alpha, beta, False, eval_func, root_player))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = float('inf')
            for move in moves:
                next_state = state.next_state(move)
                value = min(value, alphabeta(next_state, depth_left - 1, alpha, beta, True, eval_func, root_player))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    root_player = state.player
    best_move = None
    best_value = -float('inf')
    alpha = -float('inf')
    beta = float('inf')

    for move in state.legal_moves():
        next_state = state.next_state(move)
        depth_left = max_depth - 1 if max_depth > 0 else -1
        value = alphabeta(next_state, depth_left, alpha, beta, False, eval_func, root_player)
        if value > best_value:
            best_value = value
            best_move = move
        alpha = max(alpha, value)

    return best_move
