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
saidas\tabelas\06_modelos\classicos\metricas_classicos_teste.csv
saidas\tabelas\06_modelos\mobilenetv2\metricas_mobilenetv2_recortes_teste.csv
saidas\tabelas\06_modelos\comparacao\comparacao_metadados_vs_modelos_teste.csv
saidas\tabelas\07_classificacao_final\comparacao_final_classificacao.csv
saidas\tabelas\07_classificacao_final\ranking_equilibrado_classificacao.csv
saidas\tabelas\07_classificacao_final\ranking_prioridade_recall_classificacao.csv
saidas\tabelas\07_classificacao_final\resumo_comparacao_classificacao.txt
saidas\tabelas\07_classificacao_final\validacao_tratamento\resumo_generalizacao_por_tratamento.csv
saidas\tabelas\07_classificacao_final\validacao_tratamento\comparacao_split_original_vs_tratamento.csv
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
6. Fechar a comparacao cientifica da classificacao no split original.
7. Validar generalizacao por tratamento/lote com leave-one-group-out.
8. Criar tabela integrada e analisar triagem.
9. Gerar predicoes para todos os splits, calibrar thresholds e comparar scores de triagem.
10. Interpretar se ha sinal visual real ou se origem/tratamento/pasta explicam parte relevante do resultado.

## Scripts ativos de classificacao

```text
scripts\recortes\18_treinar_recortes_resnet18.py
scripts\recortes\19_avaliar_recortes_resnet18.py
scripts\recortes\20_comparar_resultados_modelos.py
scripts\recortes\21_conferir_erros_recortes.py
scripts\recortes\22_extrair_atributos_visuais_recortes.py
scripts\recortes\23_treinar_avaliar_classicos_recortes.py
scripts\recortes\24_treinar_mobilenetv2_recortes.py
scripts\recortes\25_avaliar_mobilenetv2_recortes.py
scripts\recortes\26_baseline_metadados_classificacao.py
scripts\recortes\27_comparar_classificacao_final.py
scripts\recortes\28_validacao_por_tratamento_classificacao.py
scripts\recortes\29_gerar_relatorio_classificacao_cientifica.py
```

O script 23 usa hiperparametros escolhidos por CV estratificada de 5 folds apenas no treino. A validacao fica reservada para thresholds e o teste para avaliacao final. Ele roda o conjunto `principal_normalizado` e o conjunto de sensibilidade `sensibilidade_todos_atributos`.

Os scripts 24 e 25 treinam/avaliam MobileNetV2 com `batch_size=8`, `num_workers=4`, mixed precision, `pin_memory=True`, `persistent_workers=True`, entrada `224x224` e pesos ImageNet. O treino salva o melhor checkpoint por loss de validacao.

O script 27 consolida a comparacao final de classificacao, inclui o baseline sempre-contaminada como controle, separa resultado oficial de analise de sensibilidade e escreve os rankings em `07_classificacao_final`.

O script 28 executa validacao externa por `experimento_tratamento`, usando leave-one-group-out. O split original permanece apenas como coluna de auditoria nesse experimento. Primeiro rode:

```powershell
python scripts\recortes\28_validacao_por_tratamento_classificacao.py --preflight
```

Depois execute os modelos desejados manualmente no conda:

```powershell
python scripts\recortes\28_validacao_por_tratamento_classificacao.py --modelos random_forest svm_rbf metadados
python scripts\recortes\28_validacao_por_tratamento_classificacao.py --modelos mobilenetv2 --retomar
```

O script `28_validacao_por_tratamento_classificacao.py` e um entrypoint fino. A implementacao fica no pacote `scripts\recortes\validacao_tratamento\`, com modulos separados para configuracao, dados, folds, metricas, thresholds, persistencia, modelos, controles, agregacao e runner.

O script 29 gera o relatorio cientifico final da classificacao em `docs\relatorio_classificacao_cientifica.md`, alem de tabelas derivadas, figuras e manifesto em `saidas\tabelas\07_classificacao_final\relatorio\`. Ele apenas consolida resultados existentes.

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
