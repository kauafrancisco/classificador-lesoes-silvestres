"""
Gerador de Base Simulada de Laudos Radiográficos / Periciais (Fauna Silvestre)
==============================================================================
Gera o arquivo sintético `dados/laudos_radiograficos_simulados.csv` (3.000 laudos,
43 colunas e 9 classes de diagnóstico) utilizando apenas a biblioteca padrão do Python.

Fontes de domínio utilizadas:
  - `dados/hierarquia_anatomica.json`: 444 caminhos anatômicos estruturados em
    4 níveis hierárquicos para 5 grupos taxonômicos (Mamíferos, Aves, Anfíbios,
    Serpentes e Testudines).
  - Regras clínico-periciais: morfologia, alinhamento, integridade estrutural,
    padrão/energia de trauma, achados de tecidos moles/cavidades e escores periciais.

Mecanismos de realismo clínico implementados:
  - `taxa_nao_avaliavel` (1%): substitui pontualmente campos radiográficos por
    "Não avaliável" (simulando sobreposição ou artefatos de imagem sem gerar NaN);
  - `taxa_ruido_clinico` (3%): troca o rótulo final para uma classe clinicamente
    adjacente (simulando divergência diagnóstica real em casos limítrofes).

Nota importante sobre vazamento de dados (data leakage):
  - As 7 colunas de escores periciais (`Grau_Severidade`, `Potencial_Letal` e
    `Compatibilidade_*`) são geradas para compor um laudo pericial completo, mas
    são derivadas APÓS o diagnóstico e por isso permanecem bloqueadas no `main.py`.
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

# ------------------------------------------------------------------------------
# TABELAS DE DOMÍNIO: ESPÉCIES, ORIGENS, CLASSES E DISTRIBUIÇÃO DE FREQUÊNCIA
# ------------------------------------------------------------------------------

# Espécies silvestres brasileiras compatíveis com os 5 grupos taxonômicos.
# Cada tupla define: (Nome popular e científico, peso_min_kg, peso_max_kg).
ESPECIES_POR_GRUPO = {
    "Mamíferos": [
        ("Gambá (Didelphis albiventris)", 0.8, 2.8),
        ("Capivara (Hydrochoerus hydrochaeris)", 20.0, 55.0),
        ("Tamanduá-bandeira (Myrmecophaga tridactyla)", 18.0, 40.0),
        ("Lobo-guará (Chrysocyon brachyurus)", 18.0, 30.0),
        ("Bugio (Alouatta guariba)", 4.0, 7.5),
    ],
    "Aves": [
        ("Seriema (Cariama cristata)", 1.2, 2.2),
        ("Arara-canindé (Ara ararauna)", 0.9, 1.3),
        ("Coruja-buraqueira (Athene cunicularia)", 0.14, 0.25),
        ("Carcará (Caracara plancus)", 0.8, 1.4),
        ("Tucano-toco (Ramphastos toco)", 0.5, 0.85),
    ],
    "Anfíbios": [
        ("Sapo-cururu (Rhinella diptycha)", 0.15, 0.8),
        ("Perereca-verde (Boana faber)", 0.04, 0.12),
        ("Rã-manteiga (Leptodactylus latrans)", 0.08, 0.35),
    ],
    "Serpentes": [
        ("Jiboia (Boa constrictor)", 1.5, 14.0),
        ("Jararaca (Bothrops jararaca)", 0.3, 1.8),
        ("Cascavel (Crotalus durissus)", 0.6, 3.2),
        ("Coral-verdadeira (Micrurus frontalis)", 0.1, 0.45),
    ],
    "Testudines": [
        ("Jabuti-piranga (Chelonoidis carbonaria)", 1.5, 9.0),
        ("Tigre-d'água (Trachemys dorbigni)", 0.4, 2.5),
        ("Cágado-de-barbicha (Phrynops geoffroanus)", 1.0, 4.5),
    ],
}

# Contexto administrativo de entrada do animal no centro de triagem / perícia
# (usado no cálculo pericial de compatibilidade com atropelamento).
ORIGENS_RESGATE = [
    "Rodovia",
    "Resgate Urbano",
    "Área Rural",
    "Apreensão Ibama",
]

# As 9 classes alvo de classificação radiográfica/pericial (atributo Classificacao_Lesao)
CLASSES_LESAO = [
    "Sem Alterações Radiográficas (Normal)",
    "Lesão de Tecidos Moles",
    "Luxação Articular Isolada",
    "Fratura Simples Fechada",
    "Fratura-Luxação Articular",
    "Fratura Cominutiva Exposta",
    "Trauma Cranioencefálico Grave",
    "Trauma Raquimedular (Coluna)",
    "Trauma Torácico/Celomático Severo",
]

# Probabilidades aproximadas de ocorrência de cada uma das 9 classes (soma = 1.00),
# mantendo a base moderadamente balanceada para avaliação multiclasse.
PESOS_CLASSES = [0.12, 0.13, 0.09, 0.14, 0.10, 0.11, 0.10, 0.10, 0.11]


# ------------------------------------------------------------------------------
# 1. CARREGAMENTO E CLASSIFICAÇÃO SEMÂNTICA DA HIERARQUIA ANATÔMICA
# ------------------------------------------------------------------------------

def carregar_hierarquia_anatomica(caminho_json: Path):
    """
    Lê `dados/hierarquia_anatomica.json` e retorna todos os caminhos anatômicos
    em profundidade de 4 níveis por grupo taxonômico como tuplas:
      (grupo, nivel_1, nivel_2, nivel_3, nivel_4)
    """
    with open(caminho_json, mode="r", encoding="utf-8") as f:
        return [tuple(item) for item in json.load(f)]


def classificar_papeis_anatomicos(folha):
    """
    Mapeia cada caminho anatômico de 4 níveis para um conjunto de papéis clínicos
    compatíveis ('tecido_mole', 'cavidade_visceral', 'articulacao', 'periarticular',
    'cranio', 'coluna', 'osso_geral').

    A ordem dos blocos condicionais é proposital para evitar falsos positivos
    anatômicos (por exemplo, 'Coxal' é avaliado antes da checagem de 'sacro', e
    'Tecidos Moles' é filtrado antes de estruturas ósseas).
    """
    grupo, n1, n2, n3, n4 = folha
    texto_low = f"{n1} | {n2} | {n3} | {n4}".lower()
    papeis = set()

    # 1. Estruturas de Tecidos Moles / Tecidos Adjacentes
    #    Separa órgãos vitais de cavidade (pulmão, fígado, intestinos, etc.)
    #    de tecidos moles superficiais/periféricos.
    if n2 in ("Tecidos Moles", "Tecidos Adjacentes"):
        eh_orgao_vital_cavidade = any(
            k in texto_low
            for k in (
                "pulm",
                "cardíac",
                "vasos",
                "sacos aéreos",
                "fígado",
                "hepátic",
                "esplênic",
                "estômago",
                "ventrículo",
                "intestin",
                "duodeno",
                "jejuno",
                "íleo",
                "ceco",
                "cólon",
                "reto",
                "diafragm",
                "cloaca",
                "útero",
                "rins",
                "vesícula",
                "baço",
            )
        )
        # Se não for órgão cavitário vital, entra no pool de lesões de tecidos moles periféricos
        if not eh_orgao_vital_cavidade:
            papeis.add("tecido_mole")
        # Canal vertebral também é elegível para trauma raquimedular
        if "canal vertebral" in texto_low:
            papeis.add("coluna")
        # Órgãos internos ou tecidos situados em grandes cavidades entram em cavidade_visceral
        if eh_orgao_vital_cavidade or n1 in (
            "Tórax",
            "Abdômen",
            "Cavidade Celomática",
            "Segmento Médio",
            "Segmento Caudal",
            "Pelve",
        ):
            papeis.add("cavidade_visceral")
        return papeis

    # 2. Articulações explícitas (presentes no nível 2 da taxonomia de Mamíferos)
    if n2 == "Articulações":
        papeis.add("articulacao")
        # Articulações da coluna também são elegíveis para trauma raquimedular
        if any(
            k in texto_low
            for k in ("atlanto", "intervertebra", "axis", "c3", "sacroilíaca", "sacrococcígea")
        ):
            papeis.add("coluna")
        return papeis

    # 3. Crânio (neurocrânio) — elegível para Trauma Cranioencefálico Grave e ossos em geral
    if n2 == "Ossos do Crânio" or (grupo == "Serpentes" and n3 == "Crânio"):
        papeis.add("cranio")
        papeis.add("osso_geral")
        return papeis

    # 4. Coxal / Pelve (checado ANTES de 'sacro' para não classificar a pelve como coluna)
    if n3 == "Coxal":
        papeis.add("periarticular")
        papeis.add("osso_geral")
        return papeis

    # 5. Coluna vertebral (vértebras e estruturas axiais específicas de aves/anfíbios)
    if any(
        k in texto_low
        for k in (
            "vértebra",
            "atlas",
            "axis",
            "sacro",
            "sinsacro",
            "notório",
            "pigóstilo",
            "uróstilo",
        )
    ):
        papeis.add("coluna")
        papeis.add("osso_geral")
        # Em serpentes, "Vértebras e Costelas" também protegem a cavidade celomática
        if "costela" in texto_low:
            papeis.add("cavidade_visceral")
        return papeis

    # 6. Ossos que delimitam a cavidade torácica/celomática (costelas, esterno, casco)
    if any(
        k in texto_low
        for k in ("costela", "esterno", "quilha", "fúrcula", "carapaça", "plastrão")
    ):
        papeis.add("cavidade_visceral")
        papeis.add("osso_geral")
        return papeis

    # 7. Ossos de membros, cinturas e mandíbula (marcando regiões periarticulares)
    if n1 in ("Membro Torácico", "Membro Pélvico", "Pelve") or "mandíbula" in n2.lower() or n3 == "Mandíbula":
        papeis.add("osso_geral")
        # Regiões próximas a articulações podem sofrer luxação ou fratura-luxação
        if any(
            k in texto_low
            for k in (
                "epífise",
                "escápula",
                "coracóide",
                "glenóide",
                "acetábulo",
                "cabeça",
                "colo",
                "côndilo",
                "tróclea",
                "mandíbula",
                "articular",
                "carpo",
                "tarso",
                "cárpic",
                "társic",
                "patela",
                "cintura",
                "cinturão",
            )
        ):
            papeis.add("periarticular")
        return papeis

    # 8. Demais ossos (ex.: Ossos da Face)
    papeis.add("osso_geral")
    return papeis


def definir_lateralidade(n1: str, n2: str, n3: str) -> str:
    """
    Atribui uma lateralidade anatomicamente coerente com a estrutura sorteada:
      - Estruturas ímpares/axiais (pescoço, coluna, casco, esterno) -> 'Mediana/Axial';
      - Estruturas pares (membros, costelas, pulmões, rins, mandíbula) -> 'Esquerda',
        'Direita' ou 'Bilateral';
      - Outras regiões (ex.: cabeça/cavidades) -> sorteio entre as 4 opções.
    """
    low = f"{n1} | {n2} | {n3}".lower()
    if n1 in ("Pescoço", "Segmento Cranial", "Segmento Médio", "Segmento Caudal") or any(
        k in low for k in ("vértebra", "esterno", "sacro", "sinsacro", "carapaça", "plastrão", "traqueia", "cardíac")
    ):
        return "Mediana/Axial"
    if n1 in ("Membro Torácico", "Membro Pélvico") or any(
        k in low for k in ("costela", "coxal", "pulm", "rim", "rins", "mandíbula", "escápula", "coracóide")
    ):
        return random.choices(
            ["Esquerda", "Direita", "Bilateral"], weights=[0.42, 0.42, 0.16]
        )[0]
    return random.choice(["Esquerda", "Direita", "Bilateral", "Mediana/Axial"])


def escolher_localizacao_anatomica(buckets_grupo, diagnostico: str):
    """
    Seleciona aleatoriamente um caminho anatômico de 4 níveis clinicamente compatível
    com o diagnóstico sorteado para aquele grupo taxonômico.

    Nota: grupos que não possuem categoria explícita 'Articulações' no JSON (Aves,
    Anfíbios, Serpentes e Testudines) utilizam automaticamente o bucket 'periarticular'
    como alternativa anatomicamente válida para luxações.
    """
    if diagnostico == "Sem Alterações Radiográficas (Normal)":
        pool = buckets_grupo["todos"]
    elif diagnostico == "Lesão de Tecidos Moles":
        pool = buckets_grupo["tecido_mole"]
    elif diagnostico == "Luxação Articular Isolada":
        pool = buckets_grupo["articulacao"] or buckets_grupo["periarticular"]
    elif diagnostico == "Fratura-Luxação Articular":
        pool = buckets_grupo["periarticular"] or buckets_grupo["articulacao"]
    elif diagnostico == "Trauma Cranioencefálico Grave":
        pool = buckets_grupo["cranio"]
    elif diagnostico == "Trauma Raquimedular (Coluna)":
        pool = buckets_grupo["coluna"]
    elif diagnostico == "Trauma Torácico/Celomático Severo":
        pool = buckets_grupo["cavidade_visceral"]
    else:
        # Fratura Simples Fechada / Fratura Cominutiva Exposta:
        # prioriza ossos fora de crânio e coluna para não confundir com traumas neurológicos
        pool = [
            f
            for f in buckets_grupo["osso_geral"]
            if f not in buckets_grupo["cranio"] and f not in buckets_grupo["coluna"]
        ]
        if not pool:
            pool = buckets_grupo["osso_geral"]

    return random.choice(pool)


# ------------------------------------------------------------------------------
# 2. CÁLCULO DOS ESCORES PERICIAIS DERIVADOS (GRAU DE SEVERIDADE E COMPATIBILIDADES)
# ------------------------------------------------------------------------------

def calcular_escores_periciais(reg: dict, diagnostico: str):
    """
    Calcula o Grau de Severidade (escala 0 a 5), o Potencial Letal e os 5 índices
    periciais de compatibilidade traumática a partir dos achados radiográficos limpos
    e do diagnóstico rotulado.

    Atenção: deve receber o registro ANTES da injeção de campos 'Não avaliável',
    garantindo que nenhuma regra clínica falhe silenciosamente por ausência de texto.
    """
    energia = reg["Energia_Trauma"]
    padrao = reg["Padrao_Lesao"]
    exposicao = reg["Exposicao_Lesao"]
    origem = reg["Origem_Resgate"]
    ruptura = reg["Ruptura_Tecidos_Moles"]
    exteriorizacao = reg["Exteriorizacao_Visceral"]

    # Exames normais recebem grau 0 e ausência de compatibilidade traumática
    if diagnostico == "Sem Alterações Radiográficas (Normal)":
        return {
            "Grau_Severidade": "0 - Normal",
            "Potencial_Letal": "Nulo",
            "Compatibilidade_Traumatica": "Ausente",
            "Compatibilidade_Atropelamento": "Ausente",
            "Compatibilidade_Instrumento_Cortante": "Ausente",
            "Compatibilidade_Queda": "Ausente",
            "Compatibilidade_Esmagamento": "Ausente",
        }

    # --- Definição do Grau de Severidade (1 a 5) e Potencial Letal ---
    if diagnostico == "Lesão de Tecidos Moles":
        if ruptura == "Presente" or reg["Hemorragia"] == "Presente":
            grau = "2 - Lesão leve"
            letal = "Baixo"
        else:
            grau = "1 - Alteração discreta"
            letal = "Nulo"
    elif diagnostico in ("Luxação Articular Isolada", "Fratura Simples Fechada"):
        if energia == "Baixa":
            grau = "2 - Lesão leve"
            letal = "Baixo"
        else:
            grau = "3 - Lesão moderada"
            letal = "Baixo"
    elif diagnostico == "Fratura-Luxação Articular":
        if exposicao == "Aberta" or energia == "Alta":
            grau = "4 - Lesão grave"
            letal = "Moderado"
        else:
            grau = "3 - Lesão moderada"
            letal = "Baixo"
    elif diagnostico == "Fratura Cominutiva Exposta":
        grau = "4 - Lesão grave"
        letal = "Alto" if energia == "Muito Alta" else "Moderado"
    else:
        # Traumas maiores: Cranioencefálico Grave, Raquimedular ou Torácico/Celomático Severo
        cond_incompativel = (
            reg["Integridade_Estrutural"] == "Destruída"
            or reg["Comprometimento_Neurologico"] == "Grave"
            or exteriorizacao == "Presente"
            or energia == "Muito Alta"
        )
        if cond_incompativel:
            grau = "5 - Lesão incompatível com vida"
            letal = "Crítico"
        else:
            grau = "4 - Lesão grave"
            letal = "Alto"

    # --- Definição dos 5 índices de compatibilidade pericial ---
    comp_traumatica = "Alta" if reg["Etiologia_Presumida"] == "Traumática" else "Baixa"

    # Compatibilidade com atropelamento (alta energia + lesão ampla ou origem em rodovia)
    if energia in ("Alta", "Muito Alta") and (
        reg["Distribuicao_Trauma"] in ("Multifocal", "Difusa") or origem == "Rodovia"
    ):
        comp_atropelamento = "Alta"
    elif energia in ("Moderada", "Alta", "Muito Alta"):
        comp_atropelamento = "Moderada"
    else:
        comp_atropelamento = "Baixa"

    # Compatibilidade com instrumento cortante (lesão aberta + ruptura limpa sem cominuição grave)
    if (
        exposicao == "Aberta"
        and ruptura == "Presente"
        and padrao in ("Ausente", "Fratura Linear")
        and reg["Fragmentacao"] in ("Ausente", "Leve")
    ):
        comp_cortante = "Alta"
    elif exposicao == "Aberta" and ruptura == "Presente":
        comp_cortante = "Moderada"
    else:
        comp_cortante = "Baixa"

    # Compatibilidade com queda (fraturas simples e luxações de baixa a alta energia)
    if diagnostico in (
        "Fratura Simples Fechada",
        "Luxação Articular Isolada",
        "Fratura-Luxação Articular",
    ) and energia in ("Baixa", "Moderada", "Alta"):
        comp_queda = "Alta"
    elif diagnostico in ("Lesão de Tecidos Moles", "Trauma Raquimedular (Coluna)"):
        comp_queda = "Moderada"
    else:
        comp_queda = "Baixa"

    # Compatibilidade com esmagamento (afundamento, esmagamento ou destruição/fragmentação grave)
    if padrao in ("Esmagamento", "Afundamento") or (
        reg["Integridade_Estrutural"] == "Destruída" and reg["Fragmentacao"] == "Grave"
    ):
        comp_esmagamento = "Alta"
    elif padrao == "Fratura Cominutiva" or reg["Morfologia"] == "Colapsada":
        comp_esmagamento = "Moderada"
    else:
        comp_esmagamento = "Baixa"

    return {
        "Grau_Severidade": grau,
        "Potencial_Letal": letal,
        "Compatibilidade_Traumatica": comp_traumatica,
        "Compatibilidade_Atropelamento": comp_atropelamento,
        "Compatibilidade_Instrumento_Cortante": comp_cortante,
        "Compatibilidade_Queda": comp_queda,
        "Compatibilidade_Esmagamento": comp_esmagamento,
    }


# ------------------------------------------------------------------------------
# 3. SIMULAÇÃO COMPLETA DOS LAUDOS RADIOGRÁFICOS / PERICIAIS
# ------------------------------------------------------------------------------

def gerar_laudos_periciais(
    qtd: int = 3000,
    caminho_hierarquia: Path | None = None,
    seed: int = 42,
    taxa_ruido_clinico: float = 0.03,
    taxa_nao_avaliavel: float = 0.01,
):
    """
    Gera `qtd` laudos radiográficos/periciais estruturados com base na
    hierarquia anatômica de 4 níveis (`dados/hierarquia_anatomica.json`)
    e nas regras clínico-periciais de cada uma das 9 classes diagnósticas.

    Parâmetros de realismo clínico (evitam uma base artificialmente perfeita):
      - taxa_ruido_clinico (padrão 3%): fração de casos limítrofes com sobreposição
        entre diagnósticos clinicamente adjacentes (simula divergência radiológica real).
      - taxa_nao_avaliavel (padrão 1%): fração de campos radiográficos marcados como
        'Não avaliável' (simula sobreposição/qualidade de imagem imperfeita sem
        gerar células vazias ou NaN no CSV).
    """
    # Fixa a semente para garantir que a base gerada seja 100% reproduzível
    random.seed(seed)

    base_dir = Path(__file__).resolve().parent
    if caminho_hierarquia is None:
        caminho_hierarquia = base_dir / "dados" / "hierarquia_anatomica.json"

    folhas = carregar_hierarquia_anatomica(caminho_hierarquia)

    # Indexa os 444 caminhos anatômicos por (grupo_taxonomico -> papel_clinico -> lista de folhas)
    buckets = defaultdict(lambda: defaultdict(list))
    for folha in folhas:
        grupo = folha[0]
        buckets[grupo]["todos"].append(folha)
        for papel in classificar_papeis_anatomicos(folha):
            buckets[grupo][papel].append(folha)

    grupos_disponiveis = list(ESPECIES_POR_GRUPO.keys())
    # Distribuição realista de atendimentos em centros de triagem de fauna silvestre
    # (Mamíferos 34%, Aves 34%, Anfíbios 8%, Serpentes 12%, Testudines 12%)
    pesos_grupos = [0.34, 0.34, 0.08, 0.12, 0.12]

    dados = []

    for i in range(1, qtd + 1):
        # 1) Sorteia o diagnóstico verdadeiro latente e o perfil do animal
        diagnostico = random.choices(CLASSES_LESAO, weights=PESOS_CLASSES)[0]

        # Para luxações articulares puras/mistas, serpentes têm menor incidência (apenas mandíbula)
        if diagnostico in ("Luxação Articular Isolada", "Fratura-Luxação Articular"):
            pesos_g = [0.42, 0.36, 0.09, 0.03, 0.10]
        else:
            pesos_g = pesos_grupos

        grupo = random.choices(grupos_disponiveis, weights=pesos_g)[0]
        especie, peso_min, peso_max = random.choice(ESPECIES_POR_GRUPO[grupo])
        peso = round(random.uniform(peso_min, peso_max), 2)
        origem = random.choice(ORIGENS_RESGATE)

        # 2) Sorteia uma localização anatômica de 4 níveis compatível com o diagnóstico
        _, n1, n2, n3, n4 = escolher_localizacao_anatomica(buckets[grupo], diagnostico)
        lateralidade = definir_lateralidade(n1, n2, n3)
        eh_tecido_mole = n2 in ("Tecidos Moles", "Tecidos Adjacentes")
        eh_articulacao = n2 == "Articulações"

        # 3) Inicializa todos os achados radiográficos no estado basal (animal normal)
        morfologia = "Regular"
        regularidade = "Regular"
        continuidade = "Não se aplica" if (eh_tecido_mole or eh_articulacao) else "Preservada"
        simetria = "Simétrica"
        alinhamento = "Não se aplica" if eh_tecido_mole else "Preservado"
        integridade = "Íntegra"
        fragmentacao = "Ausente"
        deslocamento = "Ausente"
        luxacao = "Ausente"
        congruencia = "Preservada" if (eh_articulacao or n1 in ("Membro Torácico", "Membro Pélvico", "Pelve")) else "Não se aplica"

        padrao_lesao = "Ausente"
        subtipo_lesao = "Não se aplica"
        etiologia = "Não se aplica"
        distribuicao = "Ausente"
        exposicao = "Não se aplica"
        energia = "Ausente"

        edema = "Ausente"
        hemorragia = "Ausente"
        contusao = "Ausente"
        enfisema = "Ausente"
        exteriorizacao = "Ausente"
        ruptura = "Ausente"
        opacidade_pulmonar = "Normal"
        hemotorax = "Ausente"
        neurologico = "Ausente"

        # ------------------------------------------------------------------
        # 4) REGRAS CLÍNICO-PERICIAIS POR CLASSE DE DIAGNÓSTICO
        # ------------------------------------------------------------------
        if diagnostico == "Sem Alterações Radiográficas (Normal)":
            # Animal resgatado pode chegar com suspeita de trauma leve sem lesão visível no RX
            etiologia = random.choices(
                ["Não informada", "Traumática", "Não traumática"], weights=[0.50, 0.35, 0.15]
            )[0]
            distribuicao = random.choices(["Ausente", "Focal"], weights=[0.65, 0.35])[0]
            energia = random.choices(["Ausente", "Baixa"], weights=[0.65, 0.35])[0]

        elif diagnostico == "Lesão de Tecidos Moles":
            # Acometimento de partes moles sem fratura óssea associada (pode ser aberta ou fechada)
            eh_aberta = random.random() < 0.45
            morfologia = random.choice(["Regular", "Irregular"])
            regularidade = "Regular" if morfologia == "Regular" else "Irregular"
            simetria = random.choice(["Simétrica", "Assimétrica"])
            integridade = "Comprometida" if eh_aberta else random.choice(["Íntegra", "Comprometida"])
            etiologia = random.choices(
                ["Traumática", "Não traumática", "Não informada"], weights=[0.75, 0.15, 0.10]
            )[0]
            distribuicao = random.choice(["Focal", "Difusa"])
            exposicao = "Aberta" if eh_aberta else "Fechada"
            energia = random.choice(["Baixa", "Moderada"])

            edema = "Presente" if random.random() < 0.85 else "Ausente"
            contusao = "Presente" if random.random() < 0.75 else "Ausente"
            enfisema = "Presente" if random.random() < 0.25 else "Ausente"
            if eh_aberta:
                ruptura = "Presente"
                hemorragia = random.choice(["Presente", "Ausente"])
            else:
                ruptura = "Ausente"
                hemorragia = "Presente" if random.random() < 0.30 else "Ausente"

            # Garante ao menos um sinal clínico de tecido mole presente para distinguir de Normal
            if all(v == "Ausente" for v in (edema, contusao, enfisema, ruptura, hemorragia)):
                contusao = "Presente"
                edema = "Presente"

        elif diagnostico == "Luxação Articular Isolada":
            # Perda de congruência articular com deslocamento, mas sem linha de fratura óssea
            morfologia = "Desviada"
            regularidade = random.choice(["Regular", "Irregular"])
            continuidade = "Não se aplica" if eh_articulacao else "Preservada"
            simetria = "Assimétrica"
            alinhamento = "Desviado"
            integridade = "Comprometida"
            fragmentacao = "Ausente"
            deslocamento = "Presente"
            luxacao = "Presente"
            congruencia = "Perdida"

            padrao_lesao = "Ausente"
            subtipo_lesao = "Não se aplica"
            etiologia = random.choices(
                ["Traumática", "Não traumática", "Não informada"], weights=[0.80, 0.12, 0.08]
            )[0]
            distribuicao = "Focal"
            exposicao = "Fechada"
            energia = random.choice(["Baixa", "Moderada", "Alta"])

            edema = random.choices(["Presente", "Ausente"], weights=[0.8, 0.2])[0]
            contusao = random.choices(["Presente", "Ausente"], weights=[0.6, 0.4])[0]
            hemorragia = random.choices(["Presente", "Ausente"], weights=[0.2, 0.8])[0]

        elif diagnostico == "Fratura Simples Fechada":
            # Fratura linear fechada sem exposição de tecidos e sem luxação associada
            continuidade = random.choices(["Parcial", "Interrompida"], weights=[0.35, 0.65])[0]
            deslocamento = random.choice(["Ausente", "Presente"])
            alinhamento = "Desviado" if deslocamento == "Presente" else "Preservado"
            morfologia = "Desviada" if deslocamento == "Presente" else "Irregular"
            regularidade = "Irregular"
            simetria = "Assimétrica"
            integridade = "Comprometida"
            fragmentacao = random.choices(["Ausente", "Leve"], weights=[0.65, 0.35])[0]
            luxacao = "Ausente"

            padrao_lesao = "Fratura Linear"
            subtipo_lesao = "Incompleta" if continuidade == "Parcial" else "Completa"
            etiologia = random.choices(
                ["Traumática", "Não traumática", "Não informada"], weights=[0.82, 0.10, 0.08]
            )[0]
            distribuicao = "Focal"
            exposicao = "Fechada"
            energia = random.choice(["Baixa", "Moderada"])

            edema = random.choices(["Presente", "Ausente"], weights=[0.75, 0.25])[0]
            contusao = random.choices(["Presente", "Ausente"], weights=[0.65, 0.35])[0]
            hemorragia = random.choices(["Presente", "Ausente"], weights=[0.25, 0.75])[0]

        elif diagnostico == "Fratura-Luxação Articular":
            # Combinação de perda de congruência articular (luxação) + fratura periarticular (ex.: Seriema)
            morfologia = random.choice(["Desviada", "Fragmentada"])
            regularidade = "Irregular"
            continuidade = "Interrompida"
            simetria = "Assimétrica"
            alinhamento = "Desviado"
            integridade = "Comprometida"
            fragmentacao = random.choice(["Leve", "Moderada"])
            deslocamento = "Presente"
            luxacao = "Presente"
            congruencia = "Perdida"

            padrao_lesao = random.choice(["Fratura Linear", "Fratura Cominutiva"])
            subtipo_lesao = random.choice(["Completa", "Múltipla"])
            etiologia = random.choices(["Traumática", "Não informada"], weights=[0.90, 0.10])[0]
            distribuicao = random.choice(["Focal", "Multifocal"])
            exposicao = random.choices(["Fechada", "Aberta"], weights=[0.70, 0.30])[0]
            energia = random.choice(["Moderada", "Alta"])

            edema = random.choices(["Presente", "Ausente"], weights=[0.9, 0.1])[0]
            contusao = random.choices(["Presente", "Ausente"], weights=[0.8, 0.2])[0]
            ruptura = "Presente" if exposicao == "Aberta" else "Ausente"
            hemorragia = "Presente" if exposicao == "Aberta" else random.choice(["Presente", "Ausente"])

        elif diagnostico == "Fratura Cominutiva Exposta":
            # Múltiplos fragmentos ósseos com exposição aberta e lesão importante de partes moles
            morfologia = "Fragmentada"
            regularidade = "Irregular"
            continuidade = "Interrompida"
            simetria = "Assimétrica"
            alinhamento = "Desviado"
            integridade = random.choice(["Comprometida", "Destruída"])
            fragmentacao = random.choice(["Moderada", "Grave"])
            deslocamento = "Presente"
            luxacao = "Ausente"

            padrao_lesao = "Fratura Cominutiva"
            subtipo_lesao = random.choice(["Completa", "Múltipla"])
            etiologia = random.choices(["Traumática", "Não informada"], weights=[0.92, 0.08])[0]
            distribuicao = random.choice(["Focal", "Multifocal"])
            exposicao = "Aberta"
            energia = random.choice(["Alta", "Muito Alta"])

            edema = "Presente"
            hemorragia = "Presente"
            contusao = "Presente"
            ruptura = "Presente"
            enfisema = random.choice(["Presente", "Ausente"])

        elif diagnostico == "Trauma Cranioencefálico Grave":
            # Trauma de alta energia em neurocrânio com destruição estrutural e déficit neurológico (ex.: Gambá)
            morfologia = random.choice(["Fragmentada", "Colapsada"])
            regularidade = "Irregular"
            continuidade = "Interrompida"
            simetria = "Assimétrica"
            alinhamento = "Desviado"
            integridade = "Destruída"
            fragmentacao = random.choice(["Moderada", "Grave"])
            deslocamento = "Presente"
            luxacao = "Ausente"
            # Permite sobreposição em congruência para evitar atalho artificial com Raquimedular
            congruencia = random.choice(["Não se aplica", "Preservada"])

            padrao_lesao = random.choice(["Afundamento", "Esmagamento", "Fratura Cominutiva"])
            subtipo_lesao = random.choice(["Completa", "Múltipla"])
            etiologia = random.choices(["Traumática", "Não informada"], weights=[0.92, 0.08])[0]
            distribuicao = random.choice(["Focal", "Multifocal", "Difusa"])
            exposicao = random.choice(["Fechada", "Aberta"])
            energia = random.choice(["Alta", "Muito Alta"])

            edema = "Presente"
            hemorragia = "Presente"
            contusao = "Presente"
            ruptura = "Presente" if exposicao == "Aberta" else random.choice(["Presente", "Ausente"])
            neurologico = random.choices(["Moderado", "Grave"], weights=[0.25, 0.75])[0]

        elif diagnostico == "Trauma Raquimedular (Coluna)":
            # Lesão vertebral/medular com desvio de eixo e comprometimento neurológico moderado a grave
            tem_luxacao = random.random() < 0.45
            morfologia = random.choice(["Irregular", "Desviada", "Fragmentada", "Colapsada"])
            regularidade = "Irregular"
            continuidade = "Não se aplica" if (eh_tecido_mole or eh_articulacao) else random.choice(["Parcial", "Interrompida"])
            simetria = "Assimétrica"
            alinhamento = "Desviado"
            integridade = random.choice(["Comprometida", "Destruída"])
            fragmentacao = "Ausente" if (eh_tecido_mole or eh_articulacao) else random.choice(["Leve", "Moderada", "Grave"])
            deslocamento = "Presente"
            luxacao = "Presente" if (eh_articulacao or tem_luxacao) else "Ausente"

            # Quando sem luxação vertebral, mantém sobreposição de congruência com Crânio
            if luxacao == "Presente":
                congruencia = "Perdida"
            else:
                congruencia = random.choice(["Preservada", "Não se aplica"])

            if eh_tecido_mole or eh_articulacao:
                padrao_lesao = "Ausente"
                subtipo_lesao = "Não se aplica"
            else:
                padrao_lesao = random.choice(["Fratura Linear", "Fratura Cominutiva", "Esmagamento"])
                subtipo_lesao = random.choice(["Completa", "Múltipla"])

            etiologia = random.choices(["Traumática", "Não informada"], weights=[0.90, 0.10])[0]
            distribuicao = random.choice(["Focal", "Multifocal"])
            exposicao = random.choices(["Fechada", "Aberta"], weights=[0.75, 0.25])[0]
            energia = random.choice(["Alta", "Muito Alta"])

            edema = random.choices(["Presente", "Ausente"], weights=[0.85, 0.15])[0]
            hemorragia = random.choices(["Presente", "Ausente"], weights=[0.75, 0.25])[0]
            contusao = random.choices(["Presente", "Ausente"], weights=[0.80, 0.20])[0]
            ruptura = "Presente" if exposicao == "Aberta" else "Ausente"
            neurologico = random.choice(["Moderado", "Grave"])

        else:
            # Trauma Torácico/Celomático Severo:
            # combina hemotórax/aumento de opacidade pulmonar e/ou exteriorização visceral
            loc_low = f"{n1} | {n3}".lower()
            eh_abdomen_ou_caudal = n1 in ("Abdômen", "Pelve", "Segmento Caudal") or any(
                k in loc_low
                for k in ("estômago", "intestin", "cloaca", "cólon", "vesícula urinária", "útero", "pró-ventrículo", "ventrículo")
            )
            if eh_abdomen_ou_caudal:
                tipo_cavidade = "exteriorizacao"
            else:
                tipo_cavidade = random.choices(
                    ["hemotorax", "exteriorizacao", "ambos"], weights=[0.65, 0.20, 0.15]
                )[0]
            com_fratura_associada = (not eh_tecido_mole) or (random.random() < 0.35)

            # Define se há comprometimento ósseo associado (ex.: costelas/casco) ou apenas visceral
            if com_fratura_associada and not eh_tecido_mole:
                morfologia = random.choice(["Irregular", "Fragmentada", "Colapsada"])
                regularidade = "Irregular"
                continuidade = "Interrompida"
                alinhamento = "Desviado"
                fragmentacao = random.choice(["Leve", "Moderada", "Grave"])
                deslocamento = "Presente"
                padrao_lesao = random.choice(["Fratura Linear", "Fratura Cominutiva", "Esmagamento"])
                subtipo_lesao = random.choice(["Completa", "Múltipla"])
            else:
                morfologia = random.choice(["Regular", "Irregular", "Colapsada"])
                regularidade = "Regular" if morfologia == "Regular" else "Irregular"
                continuidade = "Não se aplica" if eh_tecido_mole else "Preservada"
                alinhamento = "Não se aplica" if eh_tecido_mole else "Preservado"
                fragmentacao = "Ausente"
                deslocamento = "Ausente"
                padrao_lesao = "Ausente"
                subtipo_lesao = "Não se aplica"

            simetria = "Assimétrica"
            integridade = "Destruída" if tipo_cavidade in ("exteriorizacao", "ambos") else random.choice(["Comprometida", "Destruída"])
            etiologia = random.choices(["Traumática", "Não informada"], weights=[0.90, 0.10])[0]
            distribuicao = random.choice(["Focal", "Multifocal", "Difusa"])
            energia = random.choice(["Alta", "Muito Alta"])

            edema = "Presente"
            hemorragia = "Presente"
            contusao = "Presente"
            enfisema = random.choice(["Presente", "Ausente"])

            if tipo_cavidade in ("hemotorax", "ambos"):
                hemotorax = "Presente"
                opacidade_pulmonar = "Aumentada"
            if tipo_cavidade in ("exteriorizacao", "ambos"):
                exteriorizacao = "Presente"
                ruptura = "Presente"
                exposicao = "Aberta"
            else:
                exposicao = "Fechada" if not com_fratura_associada else random.choice(["Fechada", "Aberta"])
                ruptura = "Presente" if exposicao == "Aberta" else "Ausente"

        # 5) Monta o dicionário completo de entrada do laudo (35 colunas clínicas/contextuais)
        registro = {
            # Identificação e contexto do animal
            "ID_Laudo": f"RX-{i:05d}",
            "Grupo_Taxonomico": grupo,
            "Especie": especie,
            "Origem_Resgate": origem,
            "Peso_Estimado_kg": peso,
            "Lateralidade": lateralidade,
            # Localização anatômica em 4 níveis hierárquicos
            "Anatomia_Nivel_1": n1,
            "Anatomia_Nivel_2": n2,
            "Anatomia_Nivel_3": n3,
            "Anatomia_Nivel_4": n4,
            # Classificação morfológica e integridade estrutural
            "Morfologia": morfologia,
            "Regularidade": regularidade,
            "Continuidade_Cortical": continuidade,
            "Simetria": simetria,
            "Alinhamento": alinhamento,
            "Integridade_Estrutural": integridade,
            "Fragmentacao": fragmentacao,
            "Deslocamento_Osseo": deslocamento,
            "Luxacao": luxacao,
            "Congruencia_Articular": congruencia,
            # Padrão, distribuição e mecânica da lesão
            "Padrao_Lesao": padrao_lesao,
            "Subtipo_Lesao": subtipo_lesao,
            "Etiologia_Presumida": etiologia,
            "Distribuicao_Trauma": distribuicao,
            "Exposicao_Lesao": exposicao,
            "Energia_Trauma": energia,
            # Tecidos moles, cavidades e achados neurológicos
            "Edema": edema,
            "Hemorragia": hemorragia,
            "Contusao": contusao,
            "Enfisema": enfisema,
            "Exteriorizacao_Visceral": exteriorizacao,
            "Ruptura_Tecidos_Moles": ruptura,
            "Opacidade_Pulmonar": opacidade_pulmonar,
            "Hemotorax": hemotorax,
            "Comprometimento_Neurologico": neurologico,
        }

        # Guarda uma cópia com os achados limpos ANTES de injetar 'Não avaliável',
        # para que `calcular_escores_periciais` avalie sempre os achados reais do animal
        # (evitando falhas silenciosas nos `if`s quando um campo vira 'Não avaliável').
        registro_limpo = dict(registro)

        # 6) Realismo clínico — Parte A: injeção de campos 'Não avaliável' (padrão: 1% por campo)
        #    Simula sobreposição de estruturas ou limitações de posicionamento radiográfico.
        campos_radiograficos_sujeitos_a_ruido = (
            "Morfologia",
            "Regularidade",
            "Continuidade_Cortical",
            "Simetria",
            "Alinhamento",
            "Integridade_Estrutural",
            "Fragmentacao",
            "Deslocamento_Osseo",
            "Luxacao",
            "Congruencia_Articular",
            "Padrao_Lesao",
            "Subtipo_Lesao",
            "Exposicao_Lesao",
            "Edema",
            "Hemorragia",
            "Contusao",
            "Enfisema",
            "Ruptura_Tecidos_Moles",
            "Opacidade_Pulmonar",
            "Hemotorax",
            "Comprometimento_Neurologico",
        )
        if taxa_nao_avaliavel > 0:
            for campo_rx in campos_radiograficos_sujeitos_a_ruido:
                if random.random() < taxa_nao_avaliavel:
                    registro[campo_rx] = "Não avaliável"

        # 7) Realismo clínico — Parte B: ruído de rotulagem entre diagnósticos adjacentes (padrão: 3%)
        #    Simula casos limítrofes em que dois avaliadores humanos podem divergir.
        adjacencias_clinicas = {
            "Sem Alterações Radiográficas (Normal)": ["Lesão de Tecidos Moles"],
            "Lesão de Tecidos Moles": [
                "Sem Alterações Radiográficas (Normal)",
                "Fratura Simples Fechada",
            ],
            "Luxação Articular Isolada": ["Fratura-Luxação Articular"],
            "Fratura Simples Fechada": [
                "Lesão de Tecidos Moles",
                "Fratura-Luxação Articular",
            ],
            "Fratura-Luxação Articular": [
                "Luxação Articular Isolada",
                "Fratura Cominutiva Exposta",
            ],
            "Fratura Cominutiva Exposta": ["Fratura-Luxação Articular"],
            "Trauma Cranioencefálico Grave": ["Fratura Cominutiva Exposta"],
            "Trauma Raquimedular (Coluna)": ["Fratura-Luxação Articular"],
            "Trauma Torácico/Celomático Severo": ["Lesão de Tecidos Moles"],
        }
        diagnostico_rotulado = diagnostico
        if taxa_ruido_clinico > 0 and random.random() < taxa_ruido_clinico:
            diagnostico_rotulado = random.choice(adjacencias_clinicas[diagnostico])

        # 8) Calcula os 7 escores periciais usando `registro_limpo` (sem 'Não avaliável')
        #    e anexa ao registro final (bloqueados no main.py para evitar data leakage).
        escores = calcular_escores_periciais(registro_limpo, diagnostico_rotulado)
        registro.update(escores)

        # 9) Anexa a coluna alvo (Classificacao_Lesao) — totalizando 43 colunas por laudo
        registro["Classificacao_Lesao"] = diagnostico_rotulado
        dados.append(registro)

    return dados


# ------------------------------------------------------------------------------
# 4. EXPORTAÇÃO PARA CSV E EXECUÇÃO DIRETA
# ------------------------------------------------------------------------------

def salvar_csv(dados: list[dict], caminho_saida: Path):
    """Salva a lista de laudos gerados em um arquivo CSV UTF-8 com cabeçalho."""
    if not dados:
        raise ValueError("Lista de laudos vazia.")
    colunas = list(dados[0].keys())
    with open(caminho_saida, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        writer.writerows(dados)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    caminho_csv = base_dir / "dados" / "laudos_radiograficos_simulados.csv"
    laudos = gerar_laudos_periciais(qtd=3000, seed=42)
    salvar_csv(laudos, caminho_csv)
    print(f"CSV gerado com sucesso em '{caminho_csv}' ({len(laudos)} laudos, {len(laudos[0])} colunas)!")
