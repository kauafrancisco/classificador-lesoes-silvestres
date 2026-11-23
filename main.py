"""
Agente Inteligente Classificador de Lesões em Animais Silvestres
================================================================
Implementação nativa do algoritmo de Árvore de Decisão ID3 (Iterative Dichotomiser 3)
em Python puro, utilizando exclusivamente a biblioteca padrão da linguagem.

Funcionalidades principais:
1. Leitura e normalização de laudos radiográficos/periciais estruturados (CSV/JSON);
2. Divisão estratificada em conjuntos de treino (80%) e teste (20%);
3. Indução recursiva de árvore de decisão por Entropia de Shannon e Ganho de Informação
   ponderado pela cobertura de valores válidos (suporte seguro a "Não avaliável");
4. Classificação explicável (XAI) em complexidade O(d), exibindo a trilha SE-ENTÃO,
   confiança do nó e acionamento de fallback seguro com alerta de revisão manual;
5. Avaliação completa segundo o modelo PEAS (comparando ID3 Bruto vs. ID3 Seguro);
6. Ciclo de auditoria e re-treino com o especialista veterinário (modo demo e interativo).
"""

import csv
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

# Diretório-raiz do projeto (permite executar o script a partir de qualquer pasta do terminal)
BASE_DIR = Path(__file__).resolve().parent

# BLOQUEIO DE COLUNAS (PROTEÇÃO DO AGENTE)

# 1) Colunas de vazamento de dados: (Para impedir que o agente "cole")
colunas_vazamento = {
    'Grau_Severidade',
    'Potencial_Letal',
    'Compatibilidade_Traumatica',
    'Compatibilidade_Atropelamento',
    'Compatibilidade_Instrumento_Cortante',
    'Compatibilidade_Queda',
    'Compatibilidade_Esmagamento',
}

# 2) Colunas de alta cardinalidade:
colunas_alta_cardinalidade = {'Especie', 'Anatomia_Nivel_3', 'Anatomia_Nivel_4'}

# 3) Conjuntos finais de bloqueio:
colunas_block_bruto = {'Classificacao_Lesao', 'ID_Laudo', 'Peso_Estimado_kg'} | colunas_vazamento
colunas_block = colunas_block_bruto | colunas_alta_cardinalidade

# Lista ordenada das 9 classes diagnósticas oficiais
classes_lesao = [
    'Sem Alterações Radiográficas (Normal)',
    'Lesão de Tecidos Moles',
    'Luxação Articular Isolada',
    'Fratura Simples Fechada',
    'Fratura-Luxação Articular',
    'Fratura Cominutiva Exposta',
    'Trauma Cranioencefálico Grave',
    'Trauma Raquimedular (Coluna)',
    'Trauma Torácico/Celomático Severo',
]


# 1. SENSORES: LEITURA, NORMALIZAÇÃO E DIVISÃO DOS DADOS

# Normalização do registro de laudo.
def normalizar_laudo(laudo):
    laudo_limpo = {}
    for coluna, valor in laudo.items():
        if isinstance(valor, str):
            valor = valor.strip()
        if valor in ('', 'Não avaliável'): # Converte campos ausentes ou marcados como 'Não avaliável' para None
            laudo_limpo[coluna] = None
        else:
            laudo_limpo[coluna] = valor
    return laudo_limpo

# Carrega a base de dados, transformando a tabela em uma lista de dicionários, já normalizados.
def carregar_laudos(caminho_arquivo):
    with open(caminho_arquivo, mode='r', encoding='utf-8', newline='') as arquivo:
        # csv.DictReader mapeia automaticamente o cabeçalho de cada coluna para chaves do dict
        return [normalizar_laudo(linha) for linha in csv.DictReader(arquivo)]

# Separa a base em conjuntos de treino e teste, respeitando a proporção das classes.
def dividir_estratificado(laudos, proporcao_treino=0.8, semente=42):
    # Agrupa os registros por classe diagnóstica
    grupos = {}
    for laudo in laudos:
        grupos.setdefault(laudo['Classificacao_Lesao'], []).append(laudo)

    # Usa uma instância isolada de Random para não alterar o estado global do módulo random
    gerador = random.Random(semente)
    laudos_treino, laudos_teste = [], []

    # Embaralha cada classe separadamente e fatia 80% para treino e 20% para teste
    for grupo in grupos.values():
        copia = list(grupo)
        gerador.shuffle(copia)
        corte = int(len(copia) * proporcao_treino)
        laudos_treino.extend(copia[:corte])
        laudos_teste.extend(copia[corte:])

    # Embaralha os conjuntos finais para remover o agrupamento sequencial por classe
    gerador.shuffle(laudos_treino)
    gerador.shuffle(laudos_teste)
    return laudos_treino, laudos_teste



# 2. NÚCLEO DO ALGORITMO ID3

# Cálculo da Entropia de Shannon H(S):
def calcular_entropia(coluna_alvo):
    total = len(coluna_alvo)

    # Guarda de segurança contra divisão por zero em subconjuntos vazios
    if total == 0:
        return 0.0

    # Conta a frequência absoluta de cada classe no subconjunto atual
    contagem = Counter(coluna_alvo)
    entropia = 0.0
    for quantidade in contagem.values():
        probabilidade = quantidade / total
        entropia -= probabilidade * math.log2(probabilidade)
    return entropia

# Filtra o atributo que maximiza o Ganho de Informação, colocando-o no nó atual. (Obs: Aplica formulação de Quinlan em valores ausentes)
def definir_no(laudos, entropia_original, bloqueadas=None):
    if not laudos:
        return 0.0, None

    bloqueadas = colunas_block if bloqueadas is None else bloqueadas
    total_laudos = len(laudos)

    # Lista apenas os atributos ainda não utilizados no caminho da raiz e não bloqueados
    coluna_atributos = [col for col in laudos[0].keys() if col not in bloqueadas]

    maior_ganho = 0.0
    coluna_definida = None

    for coluna in coluna_atributos:
        # Filtra apenas os registros cujo valor do atributo avaliado não é None
        laudos_validos = [laudo for laudo in laudos if laudo.get(coluna) is not None]
        total_validos = len(laudos_validos)

        # Coleta os valores distintos preservando a ordem de aparição (garante determinismo)
        valores_possiveis = list(dict.fromkeys(laudo[coluna] for laudo in laudos_validos))

        # Ignora atributos totalmente nulos ou constantes (com < 2 valores não particionam nada)
        if total_validos == 0 or len(valores_possiveis) < 2:
            continue

        # Se não houve valores ausentes nesta coluna, reutiliza entropia_original já calculada, caso contrário, calcula a entropia base apenas sobre os registros válidos.
        if total_validos == total_laudos:
            entropia_base = entropia_original
        else:
            entropia_base = calcular_entropia([l['Classificacao_Lesao'] for l in laudos_validos])

        # Calcula a entropia condicional esperada após dividir pelos valores da coluna
        entropia_corte = 0.0
        for valor in valores_possiveis:
            banco_reduzido = [l for l in laudos_validos if l[coluna] == valor]
            peso = len(banco_reduzido) / total_validos
            entropia_corte += peso * calcular_entropia([l['Classificacao_Lesao'] for l in banco_reduzido])

        # Pondera o ganho pela taxa de preenchimento para penalizar atributos muito incompletos
        cobertura = total_validos / total_laudos
        ganho_informacao = cobertura * (entropia_base - entropia_corte)

        # Bloqueio de divisões "falsas"
        if ganho_informacao > maior_ganho and ganho_informacao > 1e-6:
            maior_ganho = ganho_informacao
            coluna_definida = coluna

    return maior_ganho, coluna_definida

# Indução recursiva da Árvore de Decisão, usando aninhamento de dicionários
def construir_arvore(laudos, bloqueadas=None):
    # Guarda de segurança para evitar IndexError se a função receber uma base vazia
    if not laudos:
        raise ValueError("A lista de laudos para treinamento não pode estar vazia.")

    bloqueadas = colunas_block if bloqueadas is None else bloqueadas
    total_laudos = len(laudos)

    # Identifica a classe majoritária no nó atual
    contagem = Counter(laudo['Classificacao_Lesao'] for laudo in laudos)
    classe_majoritaria, qtd_majoritaria = contagem.most_common(1)[0]
    confianca = qtd_majoritaria / total_laudos

    # Critério de parada 1: nó puro (todos os registros pertencem à mesma classe)
    if len(contagem) == 1:
        return {
            'tipo': 'folha',
            'diagnostico': classe_majoritaria,
            'confianca': 1.0,
            'amostras': total_laudos,
        }

    # Calcula a entropia do subconjunto atual e busca o melhor atributo divisor
    entropia_atual = calcular_entropia([laudo['Classificacao_Lesao'] for laudo in laudos])
    maior_ganho, melhor_coluna = definir_no(laudos, entropia_atual, bloqueadas)

    # Critério de parada 2: nenhum atributo disponível produz ganho de informação positivo
    if melhor_coluna is None or maior_ganho <= 1e-6:
        return {
            'tipo': 'folha',
            'diagnostico': classe_majoritaria,
            'confianca': confianca,
            'amostras': total_laudos,
        }

    ramos = {}
    # Propaga o atributo escolhido no conjunto de bloqueadas sem copiar os dicionários dos laudos
    proximas_bloqueadas = bloqueadas | {melhor_coluna}

    # Extrai apenas os valores válidos (exclui None para não criar ramo artificial de ausência).
    # Obs: registros com None neste atributo permanecem contabilizados em 'amostras' e 'classe_padrao' deste nó, mas não descem para as subárvores filhas.
    valores_coluna = list(dict.fromkeys(
        laudo[melhor_coluna] for laudo in laudos if laudo.get(melhor_coluna) is not None
    ))

    # Constrói recursivamente uma subárvore para cada valor observado do atributo escolhido
    for valor in valores_coluna:
        banco_reduzido = [laudo for laudo in laudos if laudo.get(melhor_coluna) == valor]
        ramos[valor] = construir_arvore(banco_reduzido, proximas_bloqueadas)

    return {
        'tipo': 'no',
        'atributo': melhor_coluna,
        'classe_padrao': classe_majoritaria,
        'confianca_padrao': confianca,
        'amostras': total_laudos,
        'ramos': ramos,
    }

# Classificação de um laudo, partindo da raiz até uma folha, equivalente ao encadeamento para frente.
def classificar_laudo(laudo_novo, arvore, limiar_confianca=0.80, caminho=None):
    # Inicializa uma nova lista de trilha a cada chamada raiz (evita bug de default mutável)
    if caminho is None:
        caminho = []

    # Caso base: chegou a um nó folha (objetivo alcançado)
    if arvore['tipo'] == 'folha':
        revisao = arvore['confianca'] < limiar_confianca
        motivo = f"Confiança abaixo de {limiar_confianca:.0%} ({arvore['confianca']:.1%})" if revisao else None
        return {
            'diagnostico': arvore['diagnostico'],
            'confianca': arvore['confianca'],
            'caminho': caminho,
            'perguntas': len(caminho),
            'revisao_manual': revisao,
            'motivo_revisao': motivo,
        }

    # Lê e normaliza o valor do atributo testado no nó interno atual
    atributo = arvore['atributo']
    resposta = laudo_novo.get(atributo)
    if isinstance(resposta, str):
        resposta = resposta.strip()
    if resposta in ('', 'Não avaliável'):
        resposta = None

    # Fallback seguro: atributo ausente ou valor não observado durante o treinamento
    if resposta is None or resposta not in arvore['ramos']:
        ausente = resposta is None
        caminho.append({
            'atributo': atributo,
            'valor': 'Não avaliável' if ausente else resposta,
            'status': 'ausente' if ausente else 'inedito',
        })
        motivo = (
            f"Atributo '{atributo}' não avaliável/ausente"
            if ausente
            else f"Valor inédito '{resposta}' para '{atributo}'"
        )
        return {
            'diagnostico': arvore['classe_padrao'],
            'confianca': arvore['confianca_padrao'],
            'caminho': caminho,
            'perguntas': len(caminho),
            'revisao_manual': True,
            'motivo_revisao': motivo,
        }

    # Passo determinístico válido: registra a condição SE percorrida e desce para o nó filho
    caminho.append({'atributo': atributo, 'valor': resposta, 'status': 'ok'})
    return classificar_laudo(laudo_novo, arvore['ramos'][resposta], limiar_confianca, caminho)


# 3. AVALIAÇÃO DE DESEMPENHO (PEAS)

# Caminho recursivo que retorna o total de nós, e o total de folhas da Árvore.
def contar_nos(arvore):
    if arvore['tipo'] == 'folha':
        return 1, 1
    nos, folhas = 1, 0
    for subarvore in arvore['ramos'].values():
        n, f = contar_nos(subarvore)
        nos += n
        folhas += f
    return nos, folhas


# Avaliação do agente sobre um conjunto rotulado de Laudos, calculando métricas do PEAS (Como acurácia, tempo de inferência, f1-score).
def avaliar_desempenho(laudos, arvore):
    inicio = time.perf_counter()
    acertos = 0
    revisoes = 0
    total_perguntas = 0

    # Inicializa a matriz de confusão com todas as classes oficiais
    matriz = {c: Counter() for c in classes_lesao}

    # Executa a inferência em todos os laudos do conjunto avaliado
    for laudo in laudos:
        real = laudo['Classificacao_Lesao']
        res = classificar_laudo(laudo, arvore)
        prev = res['diagnostico']

        # setdefault garante segurança caso surja algum rótulo fora da lista padrão
        matriz.setdefault(real, Counter())[prev] += 1
        if prev == real:
            acertos += 1
        if res['revisao_manual']:
            revisoes += 1
        total_perguntas += res['perguntas']

    tempo_ms = (time.perf_counter() - inicio) * 1000
    total = len(laudos)

    # Calcula Precisão, Recall e F1-Score individualmente para cada classe
    por_classe = {}
    f1s = []
    for classe in classes_lesao:
        tp = matriz[classe][classe]  # Verdadeiros Positivos da classe
        suporte = sum(matriz[classe].values())  # Total real de exemplos da classe
        previstos = sum(matriz[real][classe] for real in matriz)  # Total previsto como a classe

        precisao = tp / previstos if previstos else 0.0
        recall = tp / suporte if suporte else 0.0
        f1 = (2 * precisao * recall / (precisao + recall)) if (precisao + recall) else 0.0

        por_classe[classe] = {
            'suporte': suporte,
            'acertos': tp,
            'precisao': precisao,
            'recall': recall,
            'f1': f1,
        }
        f1s.append(f1)

    return {
        'total': total,
        'acertos': acertos,
        'acuracia': acertos / total if total else 0.0,
        'macro_f1': sum(f1s) / len(f1s) if f1s else 0.0,
        'revisoes': revisoes,
        'taxa_revisao': revisoes / total if total else 0.0,
        'media_perguntas': total_perguntas / total if total else 0.0,
        'tempo_ms': tempo_ms,
        'tempo_por_laudo_ms': tempo_ms / total if total else 0.0,
        'por_classe': por_classe,
        'matriz': matriz,
    }

# 4. ATUADORES: VISUALIZAÇÃO, EXPLICABILIDADE (XAI) E CICLO COM O ESPECIALISTA

# Desenho dos primeiros niveis da Árvore (Essa função foi feita pelo Gemini)
def desenhar_arvore_ascii(arvore, prefixo='', profundidade=0, max_profundidade=2):
    """
    Exibe uma prévia textual em ASCII puro dos primeiros níveis da árvore induzida,
    evitando poluir o terminal quando a árvore completa possui centenas de nós.
    """
    if arvore['tipo'] == 'folha':
        print(f"{prefixo}==> {arvore['diagnostico']} ({arvore['confianca']:.1%}, n={arvore['amostras']})")
        return

    # Imprime o atributo raiz apenas na primeira chamada (profundidade 0)
    if profundidade == 0:
        print(f"[{arvore['atributo']}?] (n={arvore['amostras']})")

    itens = list(arvore['ramos'].items())
    for i, (valor, subarvore) in enumerate(itens):
        ultimo = i == len(itens) - 1
        # Conectores ASCII compatíveis com qualquer terminal Windows/Linux
        conector = "\\-- " if ultimo else "|-- "
        extensao = "    " if ultimo else "|   "

        if subarvore['tipo'] == 'folha':
            print(f"{prefixo}{conector}{valor} -> {subarvore['diagnostico']} ({subarvore['confianca']:.1%}, n={subarvore['amostras']})")
        elif profundidade + 1 >= max_profundidade:
            # Ao atingir o limite visual, mostra o próximo atributo e a classe padrão do subnó
            print(f"{prefixo}{conector}{valor} -> [{subarvore['atributo']}?] (padrão: {subarvore['classe_padrao']}, n={subarvore['amostras']})")
        else:
            print(f"{prefixo}{conector}{valor} -> [{subarvore['atributo']}?] (n={subarvore['amostras']})")
            desenhar_arvore_ascii(subarvore, prefixo + extensao, profundidade + 1, max_profundidade)


# Exibe o caminho tomado até a decisão, junto com todas as informações (diagnóstico previsto, confiança, status de revisão)
def exibir_caminho_laudo(laudo, resultado, diagnostico_real=None):
    # Monta a localização anatômica hierárquica ignorando níveis vazios ou "Não se aplica"
    anatomia = ' > '.join(
        str(laudo.get(f'Anatomia_Nivel_{i}'))
        for i in range(1, 5)
        if laudo.get(f'Anatomia_Nivel_{i}') not in (None, '', 'Não se aplica')
    )
    print(f"Laudo {laudo.get('ID_Laudo', 'S/ID')} | {laudo.get('Grupo_Taxonomico')} - {laudo.get('Especie')}")
    print(f"  Anatomia: {anatomia}")
    print("  Regra SE-ENTÃO percorrida:")

    # Lista cada decisão tomada da raiz até o ponto de parada
    for passo_idx, passo in enumerate(resultado['caminho'], start=1):
        extra = ''
        if passo['status'] == 'ausente':
            extra = ' [ausente -> fallback do nó]'
        elif passo['status'] == 'inedito':
            extra = ' [valor inédito -> fallback do nó]'
        print(f"    {passo_idx}. SE {passo['atributo']} == '{passo['valor']}'{extra}")

    # Se houver correção do veterinário, exibe o comparativo
    if diagnostico_real is not None:
        status = 'ACERTO' if resultado['diagnostico'] == diagnostico_real else 'ERRO'
        print(f"  Diagnóstico Esperado : {diagnostico_real} [{status}]")

    print(f"  Diagnóstico Previsto : {resultado['diagnostico']} (confiança: {resultado['confianca']:.1%}, perguntas: {resultado['perguntas']})")
    if resultado['revisao_manual']:
        print(f"  Revisão Manual       : RECOMENDADA ({resultado['motivo_revisao']})")
    else:
        print("  Revisão Manual       : Não obrigatória")


# Inserção/Atualização de um laudo corrigido por veterinário (não atualiza CSV, para não corromper em testes)
def incorporar_laudo_na_base(base_treino, laudo_corrigido):
    novo = normalizar_laudo(laudo_corrigido)
    id_novo = novo.get('ID_Laudo')

    # Se o ID_Laudo já existir na base, substitui o registro antigo em vez de duplicar
    for idx, existente in enumerate(base_treino):
        if id_novo and existente.get('ID_Laudo') == id_novo:
            base_treino[idx] = novo
            return 'atualizado'

    base_treino.append(novo)
    return 'inserido'


# Ciclo para revisão do veterinário sobre a classificação final (atualmente, é opcional, para facilitar testes!!!)
def revisar_casos_clinicos(casos, base_conhecimento, arvore, interativo=False):
    confirmados, corrigidos, retreinamentos = 0, 0, 0

    try:
        for caso in casos:
            print(f"\n>>> {caso['titulo']}")
            laudo = normalizar_laudo(caso['laudo'])
            resultado = classificar_laudo(laudo, arvore)
            exibir_caminho_laudo(laudo, resultado, diagnostico_real=caso['esperado'])

            if interativo:
                # Validação para todos os tipos de resposta (ajuda de IA)
                while True:
                    resp = input("  Confirmar diagnóstico? [S/n]: ").strip().lower()
                    if resp in ('', 's', 'sim', 'n', 'nao', 'não'):
                        break
                    print("  Opção inválida. Digite 'S' para confirmar ou 'n' para corrigir.")

                if resp in ('', 's', 'sim'):
                    confirmados += 1
                    print("  [Especialista] Diagnóstico confirmado.")
                    continue

                # Exibe o menu numerado das 9 classes oficiais para correção pelo veterinário
                for idx, nome in enumerate(classes_lesao, start=1):
                    print(f"    {idx}. {nome}")
                while True:
                    entrada = input(f"  Informe o número do diagnóstico correto (1-{len(classes_lesao)}): ").strip()
                    if entrada.isdigit() and 1 <= int(entrada) <= len(classes_lesao):
                        diag_correto = classes_lesao[int(entrada) - 1]
                        break
                    print(f"  Opção inválida. Digite um número entre 1 e {len(classes_lesao)}.")
            else:
                # No modo demonstração, confirma automaticamente acertos sem pendência de revisão
                if resultado['diagnostico'] == caso['esperado'] and not resultado['revisao_manual']:
                    confirmados += 1
                    print("  [Especialista - Demo] Diagnóstico confirmado.")
                    continue
                diag_correto = caso['esperado']

            # Diferencia mensagem caso o especialista tenha alterado a classe ou apenas validado um alerta
            mudou_diagnostico = diag_correto != resultado['diagnostico']
            verbo_acao = 'Corrigido para' if mudou_diagnostico else 'Validado sob revisão como'
            corrigidos += 1

            # Incorpora o laudo rotulado na base em memória e re-treina a árvore imediatamente
            laudo_corrigido = dict(laudo, Classificacao_Lesao=diag_correto)
            acao = incorporar_laudo_na_base(base_conhecimento, laudo_corrigido)
            arvore = construir_arvore(base_conhecimento)
            retreinamentos += 1
            nos, folhas = contar_nos(arvore)

            print(
                f"  [Especialista] {verbo_acao} '{diag_correto}' -> laudo {acao} em memória "
                f"(base: {len(base_conhecimento)} | árvore re-treinada: {nos} nós, {folhas} folhas)."
            )

            # Reavalia o próprio laudo na árvore recém-treinada para transparência do efeito do re-treino
            pos_retreino = classificar_laudo(laudo, arvore)
            tem_campo_ausente = any(p['status'] == 'ausente' for p in pos_retreino['caminho'])
            if tem_campo_ausente:
                print(
                    "  [Pós re-treino] Estatísticas do nó atualizadas em memória (CSV original preservado); "
                    "revisão manual mantida porque o laudo ainda possui campo 'Não avaliável' no caminho."
                )
            else:
                print(
                    f"  [Pós re-treino] Nova predição após re-treino: '{pos_retreino['diagnostico']}' "
                    f"(confiança: {pos_retreino['confianca']:.1%})."
                )
    except (KeyboardInterrupt, EOFError):
        print("\n  [Interrompido pelo usuário] Revisão interativa encerrada antecipadamente.")

    print(f"\n--- Resumo da Auditoria dos {len(casos)} Casos Clínicos ---")
    print(f"Revisados: {confirmados + corrigidos}/{len(casos)} | Confirmados: {confirmados} | Corrigidos/Validados: {corrigidos} | Re-treinos: {retreinamentos}")
    return arvore

# 5. MAIN

if __name__ == '__main__':
    # Verifica a ativação da revisão do veterinário
    interativo = '--interativo' in sys.argv
    caminho_csv = BASE_DIR / 'dados' / 'laudos_radiograficos_simulados.csv'
    caminho_casos = BASE_DIR / 'dados' / 'casos_clinicos.json'

    # Carrega a base simulada de laudos, caso não exista, recomenda-se rodar o gerador
    try:
        laudos = carregar_laudos(caminho_csv)
    except FileNotFoundError:
        print(f"Erro: arquivo de base '{caminho_csv}' não encontrado. Favor executar 'python gerar_banco.py' primeiro!!!")
        sys.exit(1)

    # Etapa 1: Holdout 80/20 e entropia inicial calculada apenas no treino
    laudos_treino, laudos_teste = dividir_estratificado(laudos, proporcao_treino=0.8, semente=42)
    entropia_treino = calcular_entropia([l['Classificacao_Lesao'] for l in laudos_treino])

    print('1. Preparação dos Dados (Holdout Estratificado 80/20)')
    print(f"Total: {len(laudos)} laudos | Treino: {len(laudos_treino)} | Teste: {len(laudos_teste)} | Entropia inicial (treino): {entropia_treino:.4f} bits")

    # Etapa 2: Comparação PEAS entre ID3 Bruto (com alta cardinalidade aberta) e ID3 Seguro (padrão)
    print('\n2. Avaliação PEAS — Comparação ID3 Bruto vs. ID3 Seguro')
    arvore_treinada = None
    metricas_teste = None

    for nome, bloqueio in [
        ('ID3 Bruto (N3/N4 abertos)', colunas_block_bruto),
        ('ID3 Seguro (padrão)      ', colunas_block),
    ]:
        # Mede o tempo de indução da árvore em milissegundos
        t0 = time.perf_counter()
        arv = construir_arvore(laudos_treino, bloqueadas=bloqueio)
        tempo_treino_ms = (time.perf_counter() - t0) * 1000

        # Coleta tamanho estrutural e desempenho em treino e teste
        nos, folhas = contar_nos(arv)
        m_tr = avaliar_desempenho(laudos_treino, arv)
        m_te = avaliar_desempenho(laudos_teste, arv)
        gap = (m_tr['acuracia'] - m_te['acuracia']) * 100

        # Usa .get() para não falhar caso a árvore induzida seja degenerada em uma única folha
        # Esse pedaço foi alterado por IA (apenas para organização no terminal!)
        raiz = arv.get('atributo', f"Folha({arv.get('diagnostico')})")
        print(
            f"{nome} | Raiz: {raiz:16s} | Nós: {nos:4d} ({folhas:3d} folhas) | "
            f"Treino: {m_tr['acuracia']:6.2%} | Teste: {m_te['acuracia']:6.2%} (Gap: {gap:+6.2f}%) | "
            f"Macro-F1: {m_te['macro_f1']:6.2%} | Perguntas O(d): {m_te['media_perguntas']:.2f} | "
            f"Rev.: {m_te['taxa_revisao']:5.2%} | Treino: {tempo_treino_ms:5.1f} ms | "
            f"Infer.: {m_te['tempo_por_laudo_ms']:.4f} ms/laudo"
        )

        # Guarda a árvore e as métricas da configuração padrão (ID3 Seguro) para as etapas seguintes
        if bloqueio is colunas_block:
            arvore_treinada = arv
            metricas_teste = m_te

    # Etapa 3: Prévia visual dos 2 primeiros níveis da árvore induzida pelo ID3 Seguro
    print('\n3. Prévia da Árvore Induzida (ID3 Seguro — 2 primeiros níveis)')
    desenhar_arvore_ascii(arvore_treinada, max_profundidade=2)

    # Etapa 4: Detalhamento das métricas por classe diagnóstica no conjunto de teste
    print(f"\n4. Desempenho por Classe no Conjunto de Teste ({metricas_teste['total']} laudos)")
    print(f"{'Classe de Lesão':40s} | {'Sup.':>4s} | {'Prec.':>6s} | {'Rec.':>6s} | {'F1':>6s}")
    print('-' * 74)
    for classe in classes_lesao:
        c = metricas_teste['por_classe'][classe]
        print(f"{classe:40s} | {c['suporte']:4d} | {c['precisao']:6.1%} | {c['recall']:6.1%} | {c['f1']:6.1%}")
    print('-' * 74)
    print(f"{'Acurácia Global / Macro-F1':40s} | {metricas_teste['total']:4d} | {'':>6s} | {metricas_teste['acuracia']:6.2%} | {metricas_teste['macro_f1']:6.2%}")

    # Etapa 5: Auditoria com explicação (XAI) e revisão pelo especialista sobre os casos pré-configurados
    try:
        with open(caminho_casos, mode='r', encoding='utf-8') as arquivo:
            casos_clinicos = json.load(arquivo)
    except FileNotFoundError:
        print(f"\nAviso: arquivo de casos clínicos '{caminho_casos}' não encontrado.")
        sys.exit(0)

    print(f"\n=== 5. Revisão pelo Especialista nos {len(casos_clinicos)} Casos Clínicos ===")
    # Cria uma cópia rasa da base de treino para permitir incorporação em memória sem alterar laudos_treino
    base_conhecimento = list(laudos_treino)
    arvore_treinada = revisar_casos_clinicos(casos_clinicos, base_conhecimento, arvore_treinada, interativo=interativo)
