# Agente Endrick — Trabalho 2 de Inteligência Artificial

## Grupo

- Felipe Pasinato Rossoni — 587631 — Turma B
- João Kenji Suwa — 587808 — Turma B
- Pietro Pilz de Lorenzo — 588009 — Turma B
## Bibliotecas

Nenhuma biblioteca adicional é necessária além da biblioteca padrão do Python 3.12.

## Metodologia

Durante o período de realização do trabalho, diversos agentes de IA foram consultados para sugerir e explicar melhorias no minimax e na heurística customizada inicial com o propósito de criar o melhor agente possível para o torneio. Além disso, nos baseamos em algoritmos feitos para jogar othello (IAGO feito por Rosenbloom em 1982 e LOGISTELLO feito por Michale Buro em 1997). Foram implementadas diversas soluções, baseadas em minimax e mcts, além de diferentes heurísticas e valores posicionais para o tabulairo. Após essas implementações, foi realizada uma série de competições entre 9 agentes no modelo de torneio suíço, nas quais os agentes baseados em minimax se sobressaíram. Desse modo, foram escolhidos 4 agentes, que foram mais uma vez refinados e testados um contra os outros até sr decidido o melhor, que acabou por ser uma combinação de diferentes heurísticas:


  (1) Posicional (pos): máscara EVAL_TEMPLATE (como evaluate_mask).
  (2) Mobilidade Ponderada (mob): jogadas quietas (não-frontier) valem
      3x, jogadas ruidosas (frontier) valem 1x, jogadas do oponente sao
      penalizadas em -2 cada.
  (3) Fronteira (front): peças adjacentes a espacos vazios. Negativo
      no meio e final de jogo (pecas expostas para captura são ruins).
  (4) Estabilidade Total (stable): detecção iterativa de pecas que não
      podem ser capturadas, baseada em BFS de borda + propagação nos
      4 eixos (Rosenbloom 1982, Buro 1997).
  (5) Mobilidade Potencial (pot_mob): espacos vazios adjacentes a pecas
      do oponente (preve onde o jogo pode se desenvolver).
  (6) Ameaça de Quina (threat): penalidade se oponente ocupa casa X
      adjacente a quina vazia; bonus se o jogador a ocupa.
  (7) Paridade (parity): quem faz a ultima jogada (calculada nos
      ultimos 24 espacos vazios).
  (8) Penalidades X e C (x_penalty, c_penalty): penalidades por ocupar
      casas perigosas adjacentes a quinas. X são as quatro casas diagonais das quinas e C são as 8 casas adjacentes às quinas.


As fórmulas a seguir foram desenvolvidas por agentes de IA e aperfeiçoadas por nós. Nosso objetivo é variar os pesos das heurísticas a depender do momento do jogo:

  Fase (pecas)    Formula
  ------------------------------------------------------------
  <= 12 (abertura) pos + pot_mob*3 + mob*2 + piece*8 + stable*3
                    + x_penalty + threat
  13-24            pos + mob*8 - front*1 + piece*3 + stable*5
                    + parity*2 + pot_mob*2 + x_penalty + c_penalty
                    + threat
  25-40            pos + mob*14 - front*2 + piece*2 + stable*4
                    + parity*3 + pot_mob*1 + x_penalty + c_penalty
                    + threat
  41-56            pos + mob*18 - front*3 + piece*3 + stable*3
                    + parity*4 + x_penalty + c_penalty + threat
  > 56 (final)     pos + mob*6 + piece*14 + stable*2 + parity*10
                    + threat

Justificativa das decisoes de projeto:

  - Mobilidade e o fator mais importante em Othello (Buro, 1997).
    Por isso, seu peso e o maior durante o meio-jogo (até 18).

  - A contagem de peças é enganosa no inicio (peso 2-3) mas crucial
    no final (peso 14), quando o que importa e o placar. Diversos testes houve "virada" nas etapas finais do jogo, apesar de no meio do jogo o agente em questão estar perdendo por mais de 20 peças ou ter apenas uma ou duas em alguns jogos.

  - A estabilidade total e computada apenas quando ha >= 8 pecas,
    pois antes disso o cálculo é caro e pouco informativo.

  - A fronteira é negativa no meio-jogo: ter peças expostas permite
    que o oponente as capture. Isso acaba gerando o fenômeno de o agente acabar tendo poucas peças no meio e deixar o oponente tomar essas fronteiras no meio-jogo.

  - A paridade só é relevante nos últimos 24 espaços e passa a ser crucial para os cálculos.

  - As penalidades X/C sao altas (-40, -15) porque ocupar essas casas
    é quase sempre um erro grave que entrega a quina ao oponente. Essa heurística foi percebida ao jogarmos nós mesmos contra agentes pela internet como https://othello-rust.web.app/.

  - A mobilidade ponderada distingue jogadas quietas (seguras) de
    ruidosas (que expandem a fronteira), o que é mais informativo que
    a contagem de jogadas legais, pois controla mais a expansão do jogo.

Quanto ao minimax implementado, utilizamos 8 técnicas de otimização:

  (a) Aprofundamento Iterativo (ID)
    Busca em profundidades crescentes (1, 2, 3, ...). Permite parar
    quando o tempo acaba e retornar a melhor jogada da iteração
    anterior. O custo e dominado pela última iteração.

  (b) Principal Variation Search (PVS)
    PVS (tambem chamado de Negascout) e uma variante da poda alfa-beta que
    explora o primeiro filho com janela total e os demais com uma janela
    extremamente estreita, (alpha, alpha+1), chamada "janela nula" ou
    "zero-window search". A ideia central é se a ordenacao de jogadas for boa, o primeiro filho é
    o melhor na grande maioria dos casos. Os filhos seguintes são provavelmente
    piores e podem ser descartados rapidamente.

  (c) Late Move Reduction (LMR)
    LMR e uma técnica que reduz a profundidade de busca para jogadas que
    aparecem mais tarde na ordenacao (por exemplo, provavelmente piores). A ideia
    é se uma jogada aparece em 5o lugar na ordenacao, ela provavelmente é ruim e não precisa ser explorada com profundidade total para ser refutada.

  (d) Tabela de Transposição (TT)
    Dicionario com ate 300k entradas indexadas por (str(board),
    player). Cada entrada guarda valor, profundidade, flag
    (EXACT (sem corte)/LOWER (com poda beta)/UPPER (com poda alfa)) e melhor jogada. Quando cheia, remove 50% das entradas mais antigas para liberar espaço. Após o retorno da função, esse dicionário é esquecido, sendo refeito novamente durante o próximo turno, sem guardar informações entre turnos.

  (e) Janelas de Aspiracao
    Em vez de sempre iniciar a busca de cada iteracao do ID com a janela
    total (-inf, +inf), usa-se uma janela estreita centrada no valor
    obtido na iteracao anterior. A hipótese e que o valor do estado raíz
    não muda drasticamente entre iterações consecutivas. O foco da busca fica em lugares mais promissores.

  (f) Killer Moves
    Jogadas que causaram corte beta são armazenadas por nível e
    tentadas primeiro em irmãos no mesmo nível.

  (g) Heuristica de Historico
    Enquanto Killer guarda apenas uma jogada por nível, a heurística de
    histórico acumula um score para cada par (x, y, nível), ou seja, para
    cada posicçã do tabuleiro em cada nivel, sempre que essa jogada
    causa um corte beta.

  (h) Internal Iterative Deepening (IID)
    Se a Tabela de Transposição não tem informação suficiente sobre um
    estado (ou a informação está desatualizada), faz-se uma busca rasa
    com metade da profundidade para obter uma jogada candidata que
    melhora a ordenação da busca completa que viria a seguir.


Após escolhido o melhor agente, ele foi testado contra os níveis hard e very hard de https://othello-rust.web.app/ para avaliar seu comportamento, apresentando bons resultados.


## Uso de IA

Utilizamos um agente de IA (Claude/Copilot) como ferramenta auxiliar durante o desenvolvimento. O uso se deu para:

- Auxiliar no projeto e refinamento da heurística customizada (peso de cada termo, detecção de estabilidade, mobilidade ponderada)
- Sugerir e explicar técnicas de otimização da poda alfa-beta (PVS, LMR, Tabela de Transposição, aprofundamento iterativo, janelas de aspiração, killer moves, heurística de histórico, IID, solver de fim de jogo)
- Depurar casos de borda (passagem de vez, estouro de tempo, ordenação de jogadas)
- Corrigir erros.
- Auxiliar implementação.

Todo o código gerado foi revisado e validado pelo grupo. As decisões finais de projeto, arquitetura e validação experimental foram tomadas pelos integrantes do grupo.

---

## Avaliação do Minimax no Tic-Tac-Toe Misère

### (i) O minimax sempre ganha ou empata jogando contra o randomplayer?

Sim. O agente com `utility` para estados terminais (1 vitória, -1 derrota, 0 empate) executa busca completa com profundidade ilimitada. Contra um `randomplayer`, o minimax nunca perde, pois explora toda a árvore de jogo e escolhe jogadas ótimas. Como o TTTM termina em no máximo 9 jogadas, a busca é exaustiva.

### (ii) O minimax sempre empata consigo mesmo?

Sim. Quando ambos os lados jogam com minimax com profundidade ilimitada, o jogo sempre termina em empate. Ambos os jogadores evitam alinhar 3 peças, resultando em um tabuleiro cheio sem vencedor.

### (iii) O minimax não perde para você quando você usa a sua melhor estratégia?

Sim. Mesmo com a melhor estratégia humana, o minimax com busca completa encontra a jogada ótima em cada posição, nunca permitindo uma derrota. O melhor resultado possível contra ele é o empate.

---

## Othello

### Heurística Customizada (`evaluate_custom`)

A heurística combina 8 termos com pesos que variam conforme a fase do jogo (5 fases):

| Termo | Descrição |
|---|---|
| **Posicional** | Máscara de valores (EVAL_TEMPLATE) — quinas +100, pré-quinas -30, etc. |
| **Mobilidade ponderada** | Jogadas quietas (não viram peças adversárias na borda) valem 3×, jogadas ruidosas valem 1×, jogadas do oponente são penalizadas em -2 cada |
| **Fronteira** | Número de peças adjacentes a espaços vazios (negativo no meio do jogo, pois expor peças é ruim) |
| **Estabilidade total** | Detecção iterativa de peças que não podem ser capturadas: começa pelas bordas conectadas a quinas e propaga para o interior ao longo de 4 eixos |
| **Mobilidade potencial** | Quantidade de espaços vazios adjacentes às peças do oponente (indicam onde o jogo pode se desenvolver) |
| **Ameaça de quina** | Penalidade de -30 se o oponente ocupa a casa X adjacente a uma quina vazia; bônus de +20 se o jogador a ocupa |
| **Paridade** | Nos 24 espaços finais, +1/-1 dependendo de quem joga o último movimento |
| **Penalidades X/C** | Ocupar casas X (adjacentes a quinas) custa -40; casas C (adjacentes a X) custam -15 |

**Pesos por fase:**

| Fase (peças) | Destaque |
|---|---|
| ≤ 12 | Peças (×8), posicional, mobilidade potencial |
| 13–24 | Mobilidade (×8), estabilidade (×5) |
| 25–40 | Mobilidade (×14), estabilidade (×4), paridade (×3) |
| 41–56 | Mobilidade (×18), paridade (×4), fronteira (−3) |
| > 56 | Peças (×14), paridade (×10) |

**Fontes:** A detecção de estabilidade foi baseada em artigos sobre Othello (Rosenbloom 1982, Buro 1997). Os pesos foram ajustados experimentalmente com partidas-teste. A mobilidade ponderada e a mobilidade potencial foram projetadas pelo grupo.

### Critério de Parada

O agente do torneio (`tournament_agent.py`) utiliza **aprofundamento iterativo com tempo adaptativo**:

- **Abertura** (≤12 peças): 3.0s
- **Meio-jogo** (13–48 peças): 4.0s
- **Final** (>48 peças): 2.0s

O limite rígido é `time_limit + 0.3s` para evitar desclassificação. A cada 50 nós, verifica-se o deadline. Se excedido, a busca atual é abortada e a melhor jogada da iteração anterior é retornada.

Quando restam ≤12 espaços vazios, o solver de fim de jogo ativa profundidade 40 (busca exaustiva até o final).

### Resultado do Minitorneio

Partidas entre as 3 heurísticas (ida e volta, com 12 partidas no total):

| # | Agente | Pts | V | D | E | PF | PA | Saldo |
|---|---|---|---|---|---|---|---|---|
| 1 | Customizada (minimax+ID) | 15 | 5 | 1 | 0 | 259 | 125 | +134 |
| 2 | Máscara posicional | 12 | 4 | 2 | 0 | 224 | 160 | +64 |
| 3 | Contagem de peças | 9 | 3 | 3 | 0 | 195 | 189 | +6 |

A heurística customizada foi a mais bem-sucedida, vencendo todas as partidas contra contagem de peças e a maioria contra a máscara posicional.

### Implementação Escolhida para o Torneio

O `tournament_agent.py` utiliza `minimax_move_id` (versão aprimorada) — uma poda alfa-beta que inclui:

- **PVS** (Principal Variation Search): busca completa apenas no primeiro filho; nos demais usa janela nula `(alpha, alpha+1)`
- **LMR** (Late Move Reduction): reduz profundidade em 1 para jogadas a partir da 4ª em profundidade ≥ 3
- **Tabela de Transposição**: dicionário com até 300k entradas; quando cheio, remove 50% das entradas mais antigas
- **Janelas de Aspiração**: a partir da 3ª iteração do ID, busca com janela `±50` em torno do valor anterior
- **Killer Moves + História**: armazena jogadas que causaram cortes beta por ply e acumula bônus no histórico
- **IID** (Internal Iterative Deepening): se a TT não tem informação suficiente, faz busca rasa antes da completa
- **Ordenação estática**: TT best → killer → histórico → template posicional (sem instanciar estados)
- **Hard deadline**: verificação a cada 50 nós com margem de 0.3s além do tempo alocado

### Extras

Além do minimax com poda alfa-beta, implementamos:

- **MCTS** (opcional, em `mcts.py`): utilizado em versões anteriores do agente, mas o minimax com as melhorias citadas apresentou desempenho superior no torneio interno
- **Solver de fim de jogo**: quando restam ≤12 espaços vazios, busca exaustiva até o final do jogo
- **5 fases de pesos na heurística**: calibração manual dos pesos para cada fase do jogo
- **Estabilidade total**: detecção completa de peças estáveis (não apenas bordas)
