from typing import Tuple
from ..othello.gamestate import GameState
from ..othello.board import Board
from .minimax import minimax_move_id

# Mascara posicional: quinas +100, pre-quinas -30/-50, bordas 6/2, centro 3, interior 1
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

# 8 direcoes para vizinhanca; 4 eixos para estabilidade
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]
AXES = [(1, 0), (0, 1), (1, 1), (1, -1)]

# Casas perigosas adjacentes a quinas
X_SQUARES = {(1, 1), (1, 6), (6, 1), (6, 6)}
C_SQUARES = {(0, 1), (1, 0), (0, 6), (1, 7), (6, 0), (7, 1), (6, 7), (7, 6)}
CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]


#Conta pecas próprias adjacentes a espaços vazios (fronteira). Importante para defesa e mobilidade. Quanto menos, melhor para o jogador, pior para o oponente.
def _count_frontier(board, player):
    f = 0
    tiles = board.tiles
    for y in range(8):
        row = tiles[y]
        for x in range(8):
            if row[x] != player:
                continue
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8 and tiles[ny][nx] == Board.EMPTY:
                    f += 1
                    break
    return f


#Encontra pecas estaveis nas bordas por BFS a partir de cada quina.
def _edge_stable(board, player):
    stable = set()
    tiles = board.tiles
    for cx, cy in CORNERS:
        if tiles[cy][cx] != player or (cx, cy) in stable:
            continue
        chain = [(cx, cy)]
        q = [(cx, cy)]
        while q:
            x, y = q.pop(0)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8 and (nx, ny) not in stable and (nx, ny) not in chain:
                    if tiles[ny][nx] == player and (nx == 0 or nx == 7 or ny == 0 or ny == 7):
                        chain.append((nx, ny))
                        q.append((nx, ny))
        for p in chain:
            stable.add(p)
    return stable


#Verifica se há peça estável em uma das direcoes do eixo. 
#Se sim, a peça é estável nessa direção. Se chegar em vazio ou peça adversária, não é estável.
def _is_stable_in_line(board, x, y, dx, dy, player, stable_set):
    nx, ny = x + dx, y + dy
    tiles = board.tiles
    while 0 <= nx < 8 and 0 <= ny < 8:
        t = tiles[ny][nx]
        if t == Board.EMPTY or t != player:
            return False
        if (nx, ny) in stable_set:
            return True
        nx += dx
        ny += dy
    return True

#Conta todas as pecas estáveis (borda + interior por propagacao nos 4 eixos). 
# Calcula a estabilidade completa, mas só é chamado se há peças suficientes (>=8) para evitar custo desnecessário no início.    
def _count_stable_full(board, player):
    edge_stable = _edge_stable(board, player)
    stable = set(edge_stable)
    changed = True
    tiles = board.tiles
    while changed:
        changed = False
        for y in range(8):
            row = tiles[y]
            for x in range(8):
                if row[x] != player or (x, y) in stable:
                    continue
                all_axes = True
                for adx, ady in AXES:
                    if not (_is_stable_in_line(board, x, y, adx, ady, player, stable) and
                            _is_stable_in_line(board, x, y, -adx, -ady, player, stable)):
                        all_axes = False
                        break
                if all_axes:
                    stable.add((x, y))
                    changed = True
    return len(stable)


#Conta espacos vazios adjacentes a pecas do oponente (mobilidade futura). Quanto mais, melhor para o jogador, pior para o oponente.
def _potential_mobility(board, player):
    opponent = Board.opponent(player)
    squares = set()
    tiles = board.tiles
    for y in range(8):
        row = tiles[y]
        for x in range(8):
            if row[x] != opponent:
                continue
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8 and tiles[ny][nx] == Board.EMPTY:
                    squares.add((nx, ny))
    return len(squares)


#Avalia ameaca a quinas vazias baseada na ocupacao das casas X adjacentes. 
#Se o oponente tem peca em X, penaliza fortemente (pode perder a quina). 
#Se nos temos, recompensa (podemos tomar a quina).
def _corner_threat(board, player):
    opponent = Board.opponent(player)
    threat = 0
    tiles = board.tiles
    for cx, cy in CORNERS:
        if tiles[cy][cx] != Board.EMPTY:
            continue
        x_sq = (cx + (1 if cx == 0 else -1), cy + (1 if cy == 0 else -1))
        if 0 <= x_sq[0] < 8 and 0 <= x_sq[1] < 8:
            t = tiles[x_sq[1]][x_sq[0]]
            if t == opponent:
                threat -= 30   # oponente pode tomar a quina
            elif t == player:
                threat += 20   # nos podemos tomar a quina
    return threat


#Mobilidade ponderada: jogadas quietas (não fronteira) valem 3x, ruidosas 1x, oponente penalizado.
def _weighted_mobility(board, player, legal_moves_player, legal_moves_opponent):
    if not legal_moves_player:
        return -len(legal_moves_opponent)
    quiet = 0
    noisy = 0
    tiles = board.tiles
    for m in legal_moves_player:
        x, y = m
        is_frontier = False
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8 and tiles[ny][nx] == Board.EMPTY:
                is_frontier = True
                break
        if is_frontier:
            noisy += 1
        else:
            quiet += 1
    return quiet * 3 + noisy - len(legal_moves_opponent) * 2


# Heuristica customizada com 8 termos e pesos por fase (5 fases).
# Combina: posicional, mobilidade ponderada, fronteira, estabilidade,
# mobilidade potencial, ameaca de quina, paridade e penalidades X/C.
def evaluate_custom(state, player):
    board = state.board
    opponent = Board.opponent(player)

    p_pieces = board.num_pieces(player)
    o_pieces = board.num_pieces(opponent)
    total = p_pieces + o_pieces

    # Termo posicional (mascara)
    pos = 0
    tiles = board.tiles
    for y in range(8):
        row = tiles[y]
        tmpl = EVAL_TEMPLATE[y]
        for x in range(8):
            p = row[x]
            if p == player:
                pos += tmpl[x]
            elif p == opponent:
                pos -= tmpl[x]

    legal_p = list(board.legal_moves(player))
    legal_o = list(board.legal_moves(opponent))

    #Mobilidade ponderada
    mob = _weighted_mobility(board, player, legal_p, legal_o)
    #Fronteira (pecas expostas)
    front = _count_frontier(board, player) - _count_frontier(board, opponent)
    # Contagem de pecas
    piece = p_pieces - o_pieces

    # Estabilidade total (calculada apenas se ha pecas suficientes)
    stable = 0
    if total >= 8:
        stable = _count_stable_full(board, player) - _count_stable_full(board, opponent)

    # Mobilidade potencial
    pot_mob = _potential_mobility(board, player) - _potential_mobility(board, opponent)
    # Ameaça de quina
    threat = _corner_threat(board, player)

    # 8. Paridade (nos ultimos 24 espacos)
    empty = 64 - total
    parity = 0
    if empty <= 24:
        parity = 1 if empty % 2 == 1 else -1

    # 9. Penalidades X e C
    x_penalty = 0
    for x, y in X_SQUARES:
        t = tiles[y][x]
        if t == player:
            x_penalty -= 40
        elif t == opponent:
            x_penalty += 10

    c_penalty = 0
    for x, y in C_SQUARES:
        t = tiles[y][x]
        if t == player:
            c_penalty -= 15
        elif t == opponent:
            c_penalty += 5

    # Combinacao linear com pesos por fase
    if total <= 12:
        return pos + pot_mob * 3 + mob * 2 + piece * 8 + stable * 3 + x_penalty + threat
    elif total <= 24:
        return pos + mob * 8 - front * 1 + piece * 3 + stable * 5 + parity * 2 + pot_mob * 2 + x_penalty + c_penalty + threat
    elif total <= 40:
        return pos + mob * 14 - front * 2 + piece * 2 + stable * 4 + parity * 3 + pot_mob * 1 + x_penalty + c_penalty + threat
    elif total <= 56:
        return pos + mob * 18 - front * 3 + piece * 3 + stable * 3 + parity * 4 + x_penalty + c_penalty + threat
    else:
        return pos + mob * 6 + piece * 14 + stable * 2 + parity * 10 + threat


def make_move(state: GameState) -> Tuple[int, int]:
    """Ponto de entrada: minimax profundidade 4 com heuristica customizada."""
    return minimax_move_id(state, 4, evaluate_custom)
