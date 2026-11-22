# Classificador de Lesões em Animais Silvestres

Projeto da disciplina de **Inteligência Artificial** voltado à classificação automática de lesões periciais em animais silvestres por meio de aprendizado indutivo (**Árvore de Decisão**), com implementação própria em **Python**.

## Objetivo

Construir um agente inteligente capaz de:
1. Induzir uma árvore de decisão a partir de laudos radiográficos estruturados (usando entropia e ganho de informação).
2. Classificar novos laudos de forma explicável, registrando o caminho de decisões até o diagnóstico.
3. Permitir a revisão e correção do diagnóstico por um médico veterinário, realimentando a base de conhecimento.

## Estado Atual (Versão 0 - Inicial)

- Implementação base do algoritmo **ID3** em Python ([main.py](main.py)) com cálculo de entropia, ganho de informação, construção recursiva da árvore, divisão dos dados (80% treino / 20% teste) e classificação de laudos.
- Gerador de base simulada ([gerar_banco.py](gerar_banco.py)) estruturado a partir da hierarquia anatômica de 4 níveis (`dados/hierarquia_anatomica.json`) e de parâmetros periciais definidos, produzindo 3.000 laudos com 43 atributos e 9 classes de lesão.

## Requisitos e Execução

- O projeto (atualmente) está utilizando apenas a biblioteca padrão do Python, sem dependências externas. (Versão atual: Python 3.10+)

Para executar o classificador com a base atual:

```bash
python main.py
```

Para gerar novamente a base simulada (`dados/laudos_radiograficos_simulados.csv`):

```bash
python gerar_banco.py
```

## Declaração de Uso de IA generativa
Até o presente momento, o **Gemini** foi utilizado neste projeto para:
- Revisão de código, apenas na indicação de erros ou detalhes ineficazes;
- Revisão textual de documentação, evitando erros ortográficos;
- Código de geração de base de dados de exemplo, para testes ("gerar_banco.py");
- Base de consulta para modelagem do problema e do agente, através de pesquisa de material teórico, recomendações de artigos, resumos e sugestões de ideias;
- Resolução de dúvidas pontuais sobre a linguagem Python.

## Declaração de Desenvolvimento Autoral
Até o presente momento, os seguintes itens foram desenvolvidos de forma totalmente autoral:
- Arquivos: "main.py";
- Modelagem do agente inteligente.

## Próximas Etapas

- Bloqueio de atributos derivados, tratamento de campos `"Não avaliável"` e divisão estratificada.
- Explicabilidade da trilha de decisão e fluxo de revisão/correção pelo especialista veterinário.
- Experimentos comparativos (PEAS) e gráficos de desempenho.
