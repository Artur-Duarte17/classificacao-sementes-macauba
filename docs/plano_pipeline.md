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

Saidas de classificacao e triagem:

```text
saidas\tabelas\06_modelos\metadados\metricas_metadados_teste.csv
saidas\tabelas\06_modelos\classicos\atributos_visuais_recortes.csv
saidas\tabelas\06_modelos\comparacao\comparacao_metadados_vs_modelos_teste.csv
saidas\tabelas\07_classificacao_final\comparacao_final_classificacao.csv
saidas\tabelas\07_classificacao_final\conclusao_classificacao.txt
saidas\tabelas\08_triagem\tabela_integrada.csv
saidas\tabelas\08_triagem\predicoes_todos_splits.csv
saidas\tabelas\08_triagem\thresholds_triagem_recomendados.csv
saidas\tabelas\08_triagem\score_triagem_recomendado.csv
```

## Ordem recomendada

1. Preparar inventario, rotulos, tabela mestre e dataset binario.
2. Treinar/avaliar baseline com imagem inteira.
3. Gerar caixas, treinar/avaliar YOLO quando necessario.
4. Treinar/avaliar classificador com recortes.
5. Rodar baseline de metadados como parte da classificacao.
6. Fechar a comparacao cientifica da classificacao.
7. Criar tabela integrada e analisar triagem.
8. Gerar predicoes para todos os splits, calibrar thresholds e comparar scores de triagem.
9. Interpretar se ha sinal visual real ou se origem/tratamento/pasta explicam parte relevante do resultado.

## Scripts ativos de classificacao

```text
scripts\recortes\18_treinar_recortes_resnet18.py
scripts\recortes\19_avaliar_recortes_resnet18.py
scripts\recortes\20_comparar_resultados_modelos.py
scripts\recortes\21_conferir_erros_recortes.py
scripts\recortes\22_extrair_atributos_visuais_recortes.py
scripts\recortes\26_baseline_metadados_classificacao.py
```

Os scripts `23-25` e `27-29` serao adicionados nas proximas etapas para modelos classicos, MobileNetV2, comparacao final, validacao por tratamento e relatorio.

## Scripts ativos de triagem

```text
scripts\triagem\30_criar_tabela_integrada.py
scripts\triagem\31_analisar_triagem.py
scripts\triagem\32_gerar_predicoes_todos_splits.py
scripts\triagem\33_calibrar_thresholds_triagem.py
scripts\triagem\34_comparar_scores_triagem.py
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
