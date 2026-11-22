import csv
import json
import random
from collections import defaultdict
from pathlib import Path

# ==============================================================================
# GERADOR DE BASE SIMULADA DE LAUDOS RADIOGRÁFICOS / PERICIAIS (FAUNA SILVESTRE)
# Utiliza:
#   - dados/hierarquia_anatomica.json (Hierarquia anatômica de 4 níveis e 5 grupos taxonômicos)
#   - Parâmetros clínicos/periciais (Morfologia, Padrão de Lesão, Severidade 0-5 e Tecidos Moles)
# ==============================================================================

# Espécies silvestres brasileiras compatíveis com os 5 grupos taxonômicos
# Formato: (Nome científico/popular, peso_min_kg, peso_max_kg)
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

ORIGENS_RESGATE = [
    "Rodovia",
    "Resgate Urbano",
    "Área Rural",
    "Apreensão Ibama",
]

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

PESOS_CLASSES = [0.12, 0.13, 0.09, 0.14, 0.10, 0.11, 0.10, 0.10, 0.11]


def carregar_hierarquia_anatomica(caminho_json: Path):
    """
    Lê dados/hierarquia_anatomica.json e retorna todos os caminhos anatômicos
    em profundidade de 4 níveis por grupo taxonômico:
      (grupo, nivel_1, nivel_2, nivel_3, nivel_4)
    """
    with open(caminho_json, mode="r", encoding="utf-8") as f:
        return [tuple(item) for item in json.load(f)]


def classificar_papeis_anatomicos(folha):
    """Mapeia cada caminho anatômico de 4 níveis para papéis clínicos compatíveis."""
    grupo, n1, n2, n3, n4 = folha
    texto_low = f"{n1} | {n2} | {n3} | {n4}".lower()
    papeis = set()

    # 1. Tecidos Moles / Tecidos Adjacentes
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
        if not eh_orgao_vital_cavidade:
            papeis.add("tecido_mole")
        if "canal vertebral" in texto_low:
            papeis.add("coluna")
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

    # 2. Articulações explícitas (Mamíferos)
    if n2 == "Articulações":
        papeis.add("articulacao")
        if any(
            k in texto_low
            for k in ("atlanto", "intervertebra", "axis", "c3", "sacroilíaca", "sacrococcígea")
        ):
            papeis.add("coluna")
        return papeis

    # 3. Crânio (neurocrânio)
    if n2 == "Ossos do Crânio" or (grupo == "Serpentes" and n3 == "Crânio"):
        papeis.add("cranio")
        papeis.add("osso_geral")
        return papeis

    # 4. Coxal / Pelve (antes da checagem de sacro para evitar falso positivo)
    if n3 == "Coxal":
        papeis.add("periarticular")
        papeis.add("osso_geral")
        return papeis

    # 5. Coluna vertebral
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
        if "costela" in texto_low:
            papeis.add("cavidade_visceral")
        return papeis

    # 6. Ossos da cavidade torácica / celomática (costelas, esterno, carapaça, plastrão)
    if any(
        k in texto_low
        for k in ("costela", "esterno", "quilha", "fúrcula", "carapaça", "plastrão")
    ):
        papeis.add("cavidade_visceral")
        papeis.add("osso_geral")
        return papeis

    # 7. Ossos de membros / cintura / mandíbula (incluindo regiões periarticulares)
    if n1 in ("Membro Torácico", "Membro Pélvico", "Pelve") or "mandíbula" in n2.lower() or n3 == "Mandíbula":
        papeis.add("osso_geral")
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
    """Atribui lateralidade coerente com a região anatômica."""
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
    """Seleciona um caminho anatômico de 4 níveis clinicamente compatível com o diagnóstico."""
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
        # Fratura Simples Fechada / Fratura Cominutiva Exposta (ossos fora crânio/coluna neuro)
        pool = [
            f
            for f in buckets_grupo["osso_geral"]
            if f not in buckets_grupo["cranio"] and f not in buckets_grupo["coluna"]
        ]
        if not pool:
            pool = buckets_grupo["osso_geral"]

    return random.choice(pool)


def calcular_escores_periciais(reg: dict, diagnostico: str):
    """
    Calcula o Grau de Severidade (0-5) e os parâmetros periciais de compatibilidade
    traumática derivados dos achados radiográficos.
    """
    energia = reg["Energia_Trauma"]
    padrao = reg["Padrao_Lesao"]
    exposicao = reg["Exposicao_Lesao"]
    origem = reg["Origem_Resgate"]
    ruptura = reg["Ruptura_Tecidos_Moles"]
    exteriorizacao = reg["Exteriorizacao_Visceral"]

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

    # Grau de severidade (escala 0 a 5) e potencial letal
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
        # Trauma Cranioencefálico Grave, Raquimedular ou Torácico/Celomático Severo
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

    comp_traumatica = "Alta" if reg["Etiologia_Presumida"] == "Traumática" else "Baixa"

    # Compatibilidade com atropelamento
    if energia in ("Alta", "Muito Alta") and (
        reg["Distribuicao_Trauma"] in ("Multifocal", "Difusa") or origem == "Rodovia"
    ):
        comp_atropelamento = "Alta"
    elif energia in ("Moderada", "Alta", "Muito Alta"):
        comp_atropelamento = "Moderada"
    else:
        comp_atropelamento = "Baixa"

    # Compatibilidade com instrumento cortante
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

    # Compatibilidade com queda
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

    # Compatibilidade com esmagamento
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


def gerar_laudos_periciais(
    qtd: int = 3000,
    caminho_hierarquia: Path | None = None,
    seed: int = 42,
    taxa_ruido_clinico: float = 0.03,
    taxa_nao_avaliavel: float = 0.01,
):
    """
    Gera `qtd` laudos radiográficos/periciais estruturados com base na
    hierarquia anatômica de 4 níveis (dados/hierarquia_anatomica.json)
    e nos parâmetros radiográficos/periciais definidos.

    Parâmetros de realismo clínico (evitam base artificialmente perfeita):
      - taxa_ruido_clinico: fração de casos limítrofes com sobreposição entre
        diagnósticos clinicamente adjacentes (simula divergência radiológica real).
      - taxa_nao_avaliavel: fração de campos radiográficos marcados como
        'Não avaliável' (simula sobreposição/qualidade de imagem imperfeita sem
        gerar NaN no CSV).
    """
    random.seed(seed)

    base_dir = Path(__file__).resolve().parent
    if caminho_hierarquia is None:
        caminho_hierarquia = base_dir / "dados" / "hierarquia_anatomica.json"

    folhas = carregar_hierarquia_anatomica(caminho_hierarquia)

    buckets = defaultdict(lambda: defaultdict(list))
    for folha in folhas:
        grupo = folha[0]
        buckets[grupo]["todos"].append(folha)
        for papel in classificar_papeis_anatomicos(folha):
            buckets[grupo][papel].append(folha)

    grupos_disponiveis = list(ESPECIES_POR_GRUPO.keys())
    # Distribuição realista de atendimentos em centros de triagem de fauna silvestre
    pesos_grupos = [0.34, 0.34, 0.08, 0.12, 0.12]

    dados = []

    for i in range(1, qtd + 1):
        diagnostico = random.choices(CLASSES_LESAO, weights=PESOS_CLASSES)[0]

        # Para luxações puras de membro/articulação, serpentes têm menor incidência (apenas mandíbula)
        if diagnostico in ("Luxação Articular Isolada", "Fratura-Luxação Articular"):
            pesos_g = [0.42, 0.36, 0.09, 0.03, 0.10]
        else:
            pesos_g = pesos_grupos

        grupo = random.choices(grupos_disponiveis, weights=pesos_g)[0]
        especie, peso_min, peso_max = random.choice(ESPECIES_POR_GRUPO[grupo])
        peso = round(random.uniform(peso_min, peso_max), 2)
        origem = random.choice(ORIGENS_RESGATE)

        _, n1, n2, n3, n4 = escolher_localizacao_anatomica(buckets[grupo], diagnostico)
        lateralidade = definir_lateralidade(n1, n2, n3)
        eh_tecido_mole = n2 in ("Tecidos Moles", "Tecidos Adjacentes")
        eh_articulacao = n2 == "Articulações"

        # Valores padrão (estado basal normal)
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
        # REGRAS CLÍNICO-PERICIAIS
        # ------------------------------------------------------------------
        if diagnostico == "Sem Alterações Radiográficas (Normal)":
            # Animal resgatado pode ter histórico suspeito (baixa energia/focal) sem lesão no RX
            etiologia = random.choices(
                ["Não informada", "Traumática", "Não traumática"], weights=[0.50, 0.35, 0.15]
            )[0]
            distribuicao = random.choices(["Ausente", "Focal"], weights=[0.65, 0.35])[0]
            energia = random.choices(["Ausente", "Baixa"], weights=[0.65, 0.35])[0]

        elif diagnostico == "Lesão de Tecidos Moles":
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

            # Garante ao menos um sinal de tecido mole presente para distinguir de Normal
            if all(v == "Ausente" for v in (edema, contusao, enfisema, ruptura, hemorragia)):
                contusao = "Presente"
                edema = "Presente"

        elif diagnostico == "Luxação Articular Isolada":
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
            # Exemplo clínico: fratura-luxação articular (ex.: Seriema)
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
            # Exemplo clínico: trauma cranioencefálico grave (ex.: Gambá)
            morfologia = random.choice(["Fragmentada", "Colapsada"])
            regularidade = "Irregular"
            continuidade = "Interrompida"
            simetria = "Assimétrica"
            alinhamento = "Desviado"
            integridade = "Destruída"
            fragmentacao = random.choice(["Moderada", "Grave"])
            deslocamento = "Presente"
            luxacao = "Ausente"
            # Evita atalho artificial com Raquimedular: permite sobreposição de congruência
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
            # Exemplo clínico: trauma raquimedular (ex.: coluna torácica)
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
            # Evita atalho artificial com Crânio: quando sem luxação vertebral, pode ser 'Não se aplica'
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
            # Trauma Torácico/Celomático Severo
            # Inspirado no Caso 2 (Gambá — Pulmão/Hemotórax) + Exteriorização visceral (OBS3)
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

        registro = {
            # Identificação e contexto
            "ID_Laudo": f"RX-{i:05d}",
            "Grupo_Taxonomico": grupo,
            "Especie": especie,
            "Origem_Resgate": origem,
            "Peso_Estimado_kg": peso,
            "Lateralidade": lateralidade,
            # Localização anatômica em 4 níveis
            "Anatomia_Nivel_1": n1,
            "Anatomia_Nivel_2": n2,
            "Anatomia_Nivel_3": n3,
            "Anatomia_Nivel_4": n4,
            # Classificação morfológica / Integridade estrutural
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
            # Padrão e distribuição da lesão
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

        # Realismo clínico: simula pequenos percentuais de campos radiográficos
        # 'Não avaliável' (sobreposição/posicionamento) e casos limítrofes adjacentes
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

        # Diagnósticos clinicamente adjacentes para casos limítrofes reais (~3,5%)
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

        # Grau de severidade (0-5) e parâmetros periciais derivados
        escores = calcular_escores_periciais(registro, diagnostico_rotulado)
        registro.update(escores)

        # Alvo principal do classificador
        registro["Classificacao_Lesao"] = diagnostico_rotulado
        dados.append(registro)

    return dados


def salvar_csv(dados: list[dict], caminho_saida: Path):
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
