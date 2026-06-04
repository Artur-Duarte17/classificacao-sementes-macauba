# Plano da pipeline

Data da revisao: 04/06/2026.

## Objetivo

Manter uma unica pipeline para predizer/triagem de contaminacao em sementes de macauba, combinando:

- imagens iniciais;
- rotulos das planilhas;
- metadados de origem, tratamento e pasta;
- predicoes dos modelos ja treinados;
- regras operacionais de triagem.

O problema continua binario para treino e avaliacao dos modelos (`contaminada` vs `nao_contaminada`), mas a saida operacional deve ser interpretada como risco:

- `alto_risco`;
- `baixo_risco`;
- `incerto`.

## Pergunta cientifica

> A imagem inicial, isolada ou combinada com metadados do experimento, consegue estimar risco de contaminacao posterior com seguranca operacional?

Perguntas secundarias:

- O sinal visual realmente acrescenta informacao alem de origem/tratamento/pasta?
- Quais modelos maximizam recall sem destruir completamente a especificidade?
- Existe vies de lote ou tratamento explicando a predicao?
- A triagem consegue criar um grupo `baixo_risco` seguro o suficiente para liberacao automatica?

## Arquivos centrais

Entradas tabulares principais:

```text
saidas\tabelas\03_tabela_mestre\tabela_mestre.csv
saidas\tabelas\03_tabela_mestre\tabela_mestre_treinavel.csv
saidas\tabelas\04_dataset_split\divisao_treino_validacao_teste.csv
saidas\tabelas\06_modelos\baseline\predicoes_baseline_resnet18_teste.csv
saidas\tabelas\06_modelos\recortes\predicoes_recortes_resnet18_teste.csv
```

Saidas integradas atuais:

```text
saidas\tabelas\07_triagem\tabela_integrada.csv
saidas\tabelas\07_triagem\predicoes_todos_splits.csv
saidas\tabelas\07_triagem\thresholds_triagem_recomendados.csv
saidas\tabelas\07_triagem\score_triagem_recomendado.csv
saidas\tabelas\06_modelos\metadados\metricas_metadados_teste.csv
saidas\tabelas\06_modelos\comparacao\comparacao_metadados_vs_modelos_teste.csv
```

## Ordem recomendada

1. Preparar inventario, rotulos, tabela mestre e dataset binario.
2. Treinar/avaliar baseline com imagem inteira.
3. Gerar caixas, treinar/avaliar YOLO quando necessario.
4. Treinar/avaliar classificador com recortes.
5. Criar tabela integrada e analisar triagem.
6. Gerar predicoes para todos os splits e calibrar thresholds na validacao.
7. Comparar scores e baseline de metadados.
8. Interpretar se ha sinal visual real ou se origem/tratamento/pasta explicam parte relevante do resultado.

## Scripts ativos de triagem

```text
scripts\triagem\22_criar_tabela_integrada.py
scripts\triagem\23_analisar_triagem.py
scripts\triagem\24_gerar_predicoes_todos_splits.py
scripts\triagem\25_calibrar_thresholds_triagem.py
scripts\triagem\26_comparar_scores_triagem.py
scripts\triagem\27_baseline_metadados.py
```

## Decisao operacional atual

A regra conservadora continua sendo a recomendacao mais segura:

```text
alto_risco -> separar
incerto -> revisar manualmente
baixo_risco -> nao usar para liberacao automatica sem nova evidencia
```

O baseline de metadados apresentou desempenho competitivo com modelos de imagem e forte separacao por grupos de tratamento/origem/pasta. Isso sugere risco real de vies de lote/tratamento e deve ser considerado em qualquer conclusao cientifica.

## O que evitar

Evitar investir em modelos maiores antes de responder se os dados atuais sustentam generalizacao:

- ResNet maior;
- ViT;
- YOLO maior sem auditoria dos pseudo-rotulos;
- ajuste extenso de hiperparametros;
- aplicativo operacional;
- conclusao de deteccao visual direta de contaminacao.

O proximo ganho metodologico tende a vir de melhor desenho de validacao, controle de lote/tratamento e atributos interpretaveis, nao apenas de redes maiores.
