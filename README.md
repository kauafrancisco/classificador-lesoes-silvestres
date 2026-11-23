# Classificador de Lesões em Animais Silvestres

Agente inteligente baseado em **Árvore de Decisão (ID3)** implementado do zero em **Python puro** para apoio à classificação de lesões radiográficas e periciais em animais silvestres.

## Requisitos e Dependências

- **Python 3.10+** (Toda a base de código foi feita utilizando funções de bibliotecas nativas do Python)

## Estrutura de Arquivos

- `main.py` — Código principal do agente: carregamento e normalização dos laudos, divisão estratificada (80/20), algoritmo **ID3** categórico com proteção anti-vazamento e filtro de alta cardinalidade, explicabilidade do caminho de decisão (*SE–ENTÃO*), métricas de desempenho (**PEAS**) e ciclo de revisão/re-treino com o especialista veterinário.
- `gerar_banco.py` — Gerador da base simulada de laudos radiográficos.
- `dados/laudos_radiograficos_simulados.csv` — Base de dados com 3.000 laudos, 43 atributos e 9 classes de lesão.
- `dados/hierarquia_anatomica.json` — Hierarquia anatômica de 4 níveis usada na geração da base.
- `dados/casos_clinicos.json` — 5 casos clínicos pré-configurados (`DEMO-001` a `DEMO-005`) para demonstração da explicabilidade e da revisão pelo veterinário.

## Como Executar

0. *(Opcional)* Para gerar novamente a base simulada (`dados/laudos_radiograficos_simulados.csv`):

```bash
python gerar_banco.py
```

1. No terminal, acesse a pasta do projeto:

```bash
cd Classificador_Lesoes_Silvestres
```

2. Para executar o agente no modo padrão (comparativo PEAS entre ID3 Bruto e ID3 Seguro, prévia ASCII da árvore, tabela de métricas por classe e demonstração dos 5 casos clínicos):

```bash
python main.py
```

3. Para ativar a revisão e correção manual dos 5 casos clínicos diretamente pelo terminal:

```bash
python main.py --interativo
```

## Declaração de Uso de IA Generativa

A ferramenta de IA generativa **Gemini** foi utilizada de forma transparente como apoio neste projeto nas seguintes frentes:

- Geração da base sintética de testes (`gerar_banco.py`) e conversão da hierarquia anatômica (`dados/hierarquia_anatomica.json`) com base nos parâmetros definidos para o domínio;
- Revisão de código, identificação de problemas (vazamento de dados) e apoio nos refinamentos do `main.py` (desenho da árvore, auxilio na visualização, reposicionamento de dados no terminal, deduplicação por `ID_Laudo` e revisão dos casos em `dados/casos_clinicos.json`);
- Apoio na pesquisa de referências teóricas (comparativos entre ID3 e C4.5 para trabalhos futuros), revisão textual, e dúvidas pontuais de Python.

### Desenvolvimento Autoral

Foram conduzidos de forma autoral pelo estudante:

- Definição do problema a partir da demanda real da equipe veterinária forense, recorte do escopo para a disciplina e organização dos parâmetros clínicos e anatômicos;
- Modelagem do agente inteligente (ambiente discreto, PEAS e fluxo entre indução, explicação e revisão pelo especialista);
- Implementação do pré-procesamento dos dados, motor do algoritmo ID3, ciclo de correção do veterinário (especialista), e avaliação (PEAS).
