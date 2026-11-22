import csv
import math
import random
from collections import Counter

# Versão 1: Totalmente "crua", fiz diretamente apenas para testar. (Base atual: Algortimo ID3)

colunas_block = {'Classificacao_Lesao', 'ID_Laudo', 'Peso_Estimado_kg'}

def calcular_entropia(coluna_alvo):
    total = len(coluna_alvo)
    contagem = Counter(coluna_alvo)

    entropia = 0.0
    for quantidade in contagem.values():
        probabilidade = quantidade / total
        entropia -= probabilidade * math.log2(probabilidade)

    return entropia

def definir_no(laudos, entropia_original):
    total_laudos = len(laudos)

    coluna_atributos = [col for col in laudos[0].keys() if col not in colunas_block]

    maior_ganho = -1
    coluna_definida = None

    for coluna in coluna_atributos:
        valores_possiveis = list(dict.fromkeys(laudo[coluna] for laudo in laudos))

        entropia_corte = 0.0

        for valor in valores_possiveis:
            banco_reduzido = [laudo for laudo in laudos if laudo[coluna] == valor]

            peso = len(banco_reduzido) / total_laudos

            entropia_reduzida = calcular_entropia([laudo['Classificacao_Lesao'] for laudo in banco_reduzido])

            entropia_corte += peso * entropia_reduzida

        ganho_informacao = entropia_original - entropia_corte

        if ganho_informacao > maior_ganho:
            maior_ganho = ganho_informacao
            coluna_definida = coluna

    return maior_ganho, coluna_definida

def construir_arvore(laudos):
    classificacoes_unicas = list(dict.fromkeys(laudo['Classificacao_Lesao'] for laudo in laudos))
    if len(classificacoes_unicas) == 1:
        return classificacoes_unicas[0] # Todos os laudos tem o mesmo diagnóstico

    entropia_atual = calcular_entropia([laudo['Classificacao_Lesao'] for laudo in laudos])

    maior_ganho, melhor_coluna = definir_no(laudos, entropia_atual)

    if melhor_coluna is None:
        return Counter(laudo['Classificacao_Lesao'] for laudo in laudos).most_common(1)[0][0]

    arvore = {melhor_coluna: {}}

    valores_coluna = list(dict.fromkeys(laudo[melhor_coluna] for laudo in laudos))

    for valor in valores_coluna:
        banco_reduzido = [
            {chave: val for chave, val in laudo.items() if chave != melhor_coluna}
            for laudo in laudos
            if laudo[melhor_coluna] == valor
        ]

        arvore_reduzida = construir_arvore(banco_reduzido)

        arvore[melhor_coluna][valor] = arvore_reduzida

    return arvore

def classificar_laudo(laudo_novo, arvore):
    if not isinstance(arvore, dict):
        return arvore

    atributo_atual = list(arvore.keys())[0]

    resposta = laudo_novo.get(atributo_atual)

    proximo_estado = arvore[atributo_atual].get(resposta)

    if proximo_estado is None:
        return f"Dados incompletos!"

    return classificar_laudo(laudo_novo, proximo_estado)

if __name__ == "__main__":
    caminho_arquivo = 'dados/laudos_radiograficos_simulados.csv'

    try:
        with open(caminho_arquivo, mode='r', encoding='utf-8', newline='') as arquivo:
            laudos = list(csv.DictReader(arquivo))
    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        exit()

    print("--- 1. Preparação de Dados (Holdout 80/20) ---")
    # Embaralha os dados para evitar vieses de ordem
    laudos_embaralhados = list(laudos)
    random.seed(42)
    random.shuffle(laudos_embaralhados)

    # Calcula o ponto de corte (80% para treino, 20% para teste)
    ponto_de_corte = int(len(laudos_embaralhados) * 0.8)

    # Faz a divisão
    laudos_treino = laudos_embaralhados[:ponto_de_corte]
    laudos_teste = laudos_embaralhados[ponto_de_corte:]

    print(f"Total de laudos originais: {len(laudos)}")
    print(f"Laudos isolados para Aprendizado (Treino): {len(laudos_treino)}")
    print(f"Laudos isolados para Desempenho (Teste)  : {len(laudos_teste)}")

    print("\n--- Teste de Entropia ---")
    entropia_total = calcular_entropia([laudo['Classificacao_Lesao'] for laudo in laudos])
    print(f"Entropia Inicial encontrada: {entropia_total:.4f}")

    print("\n--- Teste de Construção da Árvore ---")
    arvore_treinada = construir_arvore(laudos_treino)
    print("Árvore induzida com sucesso!")

    print("\n--- Elemento de Desempenho ---")
    acertos = 0
    testes_visuais = 5

    for i in range(testes_visuais):
        laudo_inedito = laudos_teste[i]
        diagnostico_real = laudo_inedito['Classificacao_Lesao']

        diagnostico_ia = classificar_laudo(laudo_inedito, arvore_treinada)

        status = "ACERTO" if diagnostico_ia == diagnostico_real else "ERRO"
        if diagnostico_ia == diagnostico_real: acertos += 1

        print(f"Laudo {i+1}:")
        print(f"  Diagnóstico Real do Veterinário: {diagnostico_real}")
        print(f"  Diagnóstico Previsto pela IA   : {diagnostico_ia}")
        print(f"  Status: {status}\n")
