import time
from typing import Tuple, Callable

# Flags da Tabela de Transposicao
EXACT = 0      # valor exato do no
LOWER = 1      # corte beta: valor real >= armazenado
UPPER = 2      # corte alfa: valor real <= armazenado

TT_MAX = 300000           # maximo de entradas na TT antes de podar
HARD_LIMIT_MARGIN = 0.3   # margem extra de segundos apos o deadline
MINIMAX_HISTORY_MAX = 1 << 20  # teto para o acumulador de historico

ORDER_TEMPLATE = [
    [100, -30, 6, 2, 2, 6, -30, 100],
    [-30, -50, 1, 1, 1, 1, -50, -30],
    [  6,   1, 1, 1, 1, 1,   1,   6],
    [  2,   1, 1, 3, 3, 1,   1,   2],
    [  2,   1, 1, 3, 3, 1,   1,   2],
    [  6,   1, 1, 1, 1, 1,   1,   6],
    [-30, -50, 1, 1, 1, 1, -50, -30],
    [100, -30, 6, 2, 2, 6, -30, 100]
]

#Ordena jogadas: TT best > killer/IID > historico + template posicional.
def order_moves_fast(moves, tt_best, pv_move, history, ply):
    scored = []
    for m in moves:
        priority = 0
        if tt_best is not None and m == tt_best:
            priority = 10_000_000
        elif pv_move is not None and m == pv_move:
            priority = 9_000_000
        else:
            priority = history.get((m[0], m[1], ply), 0)
            if 0 <= m[1] < 8 and 0 <= m[0] < 8:
                priority += ORDER_TEMPLATE[m[1]][m[0]] * 10
        scored.append((-priority, m))
    scored.sort(key=lambda x: x[0])
    return [m for _, m in scored]


#Minimax generico com poda alfa-beta, profundidade fixa. Para TTTM ou Othello.
def minimax_move(state, max_depth: int, eval_func: Callable) -> Tuple[int, int]:
    root_player = state.player

    def search(current_state, depth, alpha, beta, maximizing):
        # Folha: terminal ou profundidade esgotada
        if current_state.is_terminal():
            return eval_func(current_state, root_player), None
        if max_depth != -1 and depth <= 0:
            return eval_func(current_state, root_player), None

        legal = list(current_state.legal_moves())

        # Passagem de vez (sem jogadas legais)
        if not legal:
            ns = current_state.next_state(None)
            next_is_maximizing = (ns.player == root_player)
            val, _ = search(ns, depth - 1, alpha, beta, next_is_maximizing)
            return val, None

        best_move = None
        if maximizing:
            value = float('-inf')
            for move in legal:
                ns = current_state.next_state(move)
                next_is_maximizing = (ns.player == root_player)
                val, _ = search(ns, depth - 1, alpha, beta, next_is_maximizing)
                if val > value:
                    value = val
                    best_move = move
                alpha = max(alpha, value)
                if value >= beta:
                    break   # poda beta
            return value, best_move
        else:
            value = float('inf')
            for move in legal:
                ns = current_state.next_state(move)
                next_is_maximizing = (ns.player == root_player)
                val, _ = search(ns, depth - 1, alpha, beta, next_is_maximizing)
                if val < value:
                    value = val
                    best_move = move
                beta = min(beta, value)
                if value <= alpha:
                    break   # poda alfa
            return value, best_move

    initial_depth = max_depth if max_depth != -1 else 8
    _, move = search(state, initial_depth, float('-inf'), float('inf'), True)
    return move


def _state_key(state):
    """Chave unica para a Tabela de Transposicao."""
    return (str(state.board), state.player)


def _should_solve_endgame(state):
    """Ativa solver exaustivo se restam <= 12 casas vazias."""
    board = state.board
    total = board.num_pieces('B') + board.num_pieces('W')
    return 64 - total <= 12


def _prune_tt(tt, max_size):
    """Remove 50% das entradas mais antigas da TT quando excede o limite."""
    if len(tt) > max_size:
        keys = list(tt.keys())
        for k in keys[:len(keys) // 2]:
            del tt[k]

#Minimax com aprofundamento iterativo + PVS + LMR + TT + killers + historico + IID (Othello).
def minimax_move_id(state, max_depth: int, eval_func: Callable, time_limit: float = None) -> Tuple[int, int]:
    root_player = state.player
    moves = list(state.legal_moves())
    if not moves:
        return None
    if len(moves) == 1:
        return moves[0]

    if time_limit is None:
        time_limit = 4.0

    solve = _should_solve_endgame(state)
    max_search_depth = 40 if solve else (max_depth if max_depth > 0 else 40)

    best_move = moves[0]
    start = time.time()
    hard_limit = time_limit + HARD_LIMIT_MARGIN
    nodes = 0
    deadline = False
    tt = {}
    history = {}
    killers = {}

    def search_pvs(current_state, depth, alpha, beta, maximizing, ply):
        # PVS com LMR, TT, killers e IID
        nonlocal nodes, deadline

        nodes += 1
        if nodes % 50 == 0 and time.time() - start >= hard_limit:
            deadline = True

        if deadline:
            return (eval_func(current_state, root_player) if maximizing
                    else -eval_func(current_state, root_player)), None

        if current_state.is_terminal():
            return eval_func(current_state, root_player), None
        if depth <= 0:
            return eval_func(current_state, root_player), None

        legal = list(current_state.legal_moves())

        if not legal:
            ns = current_state.next_state(None)
            next_is_maximizing = (ns.player == root_player)
            val, _ = search_pvs(ns, depth - 1, alpha, beta, next_is_maximizing, ply + 1)
            return val, None

        # Consulta a Tabela de Transposicao
        key = _state_key(current_state)
        tt_entry = tt.get(key)
        if tt_entry is not None and tt_entry['depth'] >= depth:
            if tt_entry['flag'] == EXACT:
                return tt_entry['value'], tt_entry.get('move')
            elif tt_entry['flag'] == LOWER:
                alpha = max(alpha, tt_entry['value'])
            elif tt_entry['flag'] == UPPER:
                beta = min(beta, tt_entry['value'])
            if alpha >= beta:
                return tt_entry['value'], tt_entry.get('move')

        tt_best = tt_entry.get('move') if tt_entry is not None else None

        # Internal Iterative Deepening: busca rasa se TT nao tem info
        iid_move = None
        if tt_entry is None or tt_entry['depth'] < depth - 2:
            if depth >= 4:
                _, iid_move = search_pvs(current_state, depth // 2, alpha, beta, maximizing, ply)

        pv_move = killers.get(ply) if iid_move is None else iid_move
        ordered = order_moves_fast(legal, tt_best, pv_move, history, ply)
        best_local = None
        moves_tried = 0

        if maximizing:
            value = float('-inf')
            for move in ordered:
                if deadline:
                    break
                ns = current_state.next_state(move)
                next_is_maximizing = (ns.player == root_player)
                moves_tried += 1

                # Primeiro filho: busca completa; demais: janela nula + LMR
                if moves_tried == 1:
                    val, _ = search_pvs(ns, depth - 1, alpha, beta, next_is_maximizing, ply + 1)
                else:
                    r = 0
                    if depth >= 3 and moves_tried >= 4:
                        r = 1   # LMR: reduz profundidade em 1
                    new_depth = depth - 1 - r
                    if new_depth > 0:
                        val, _ = search_pvs(ns, new_depth, alpha, alpha + 1, next_is_maximizing, ply + 1)
                        if val > alpha and val < beta and not deadline:
                            val, _ = search_pvs(ns, depth - 1, alpha, beta, next_is_maximizing, ply + 1)
                    else:
                        val, _ = search_pvs(ns, depth - 1, alpha, alpha + 1, next_is_maximizing, ply + 1)
                        if val > alpha and val < beta and not deadline:
                            val, _ = search_pvs(ns, depth - 1, alpha, beta, next_is_maximizing, ply + 1)

                if val > value:
                    value = val
                    best_local = move
                alpha = max(alpha, value)
                if value >= beta:
                    if best_local is not None:
                        killers[ply] = best_local
                        hkey = (best_local[0], best_local[1], ply)
                        history[hkey] = min(history.get(hkey, 0) + (1 << depth), MINIMAX_HISTORY_MAX)
                    tt[key] = {'value': value, 'depth': depth, 'flag': LOWER, 'move': best_local}
                    return value, best_local

            tt[key] = {'value': value, 'depth': depth, 'flag': EXACT if best_local is not None else UPPER, 'move': best_local}
            return value, best_local
        else:
            value = float('inf')
            for move in ordered:
                if deadline:
                    break
                ns = current_state.next_state(move)
                next_is_maximizing = (ns.player == root_player)
                moves_tried += 1

                if moves_tried == 1:
                    val, _ = search_pvs(ns, depth - 1, alpha, beta, next_is_maximizing, ply + 1)
                else:
                    r = 0
                    if depth >= 3 and moves_tried >= 4:
                        r = 1
                    new_depth = depth - 1 - r
                    if new_depth > 0:
                        val, _ = search_pvs(ns, new_depth, beta - 1, beta, next_is_maximizing, ply + 1)
                        if val < beta and val > alpha and not deadline:
                            val, _ = search_pvs(ns, depth - 1, alpha, beta, next_is_maximizing, ply + 1)
                    else:
                        val, _ = search_pvs(ns, depth - 1, beta - 1, beta, next_is_maximizing, ply + 1)
                        if val < beta and val > alpha and not deadline:
                            val, _ = search_pvs(ns, depth - 1, alpha, beta, next_is_maximizing, ply + 1)

                if val < value:
                    value = val
                    best_local = move
                beta = min(beta, value)
                if value <= alpha:
                    if best_local is not None:
                        killers[ply] = best_local
                        hkey = (best_local[0], best_local[1], ply)
                        history[hkey] = min(history.get(hkey, 0) + (1 << depth), MINIMAX_HISTORY_MAX)
                    tt[key] = {'value': value, 'depth': depth, 'flag': UPPER, 'move': best_local}
                    return value, best_local

            tt[key] = {'value': value, 'depth': depth, 'flag': EXACT if best_local is not None else LOWER, 'move': best_local}
            return value, best_local

    prev_eval = None
    for depth in range(1, max_search_depth + 1):
        if time.time() - start >= time_limit or deadline:
            break
        _prune_tt(tt, TT_MAX)

        # Janela de aspiracao a partir da 3a iteracao
        if prev_eval is not None and depth >= 3:
            window = 50
            alpha_asp = prev_eval - window
            beta_asp = prev_eval + window
            _, move = search_pvs(state, depth, alpha_asp, beta_asp, True, 0)
            if deadline:
                break
            if move is None:
                _, move = search_pvs(state, depth, float('-inf'), float('inf'), True, 0)
        else:
            _, move = search_pvs(state, depth, float('-inf'), float('inf'), True, 0)
            if deadline:
                break

        if move is not None and not deadline:
            best_move = move
            prev_eval = None

    return best_move
