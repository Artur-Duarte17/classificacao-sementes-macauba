# Relatorio da triagem preventiva

Gerado em: 2026-06-08T15:38:17

## 1. Objetivo

A triagem preventiva transforma os resultados de classificacao em uma regra
operacional conservadora. Ela nao substitui a avaliacao manual: o objetivo e
priorizar alto risco, manter incerteza quando o sinal nao e suficiente e evitar
tratar baixo risco como liberacao automatica.

## 2. Entradas

- `saidas\tabelas\08_triagem\tabela_integrada_triagem.csv`
- `saidas\tabelas\08_triagem\scores_candidatos_triagem.csv`
- `saidas\tabelas\08_triagem\thresholds_crossfit_por_grupo.csv`
- `saidas\tabelas\08_triagem\predicoes_triagem_crossfit.csv`
- `saidas\tabelas\08_triagem\metricas_triagem_por_grupo.csv`
- `saidas\tabelas\08_triagem\resumo_triagem_micro_macro.csv`
- `saidas\tabelas\08_triagem\score_triagem_recomendado.csv`

## 3. Estrategia oficial

A estrategia oficial foi definida antes da avaliacao externa:
`consenso_pre_especificado`.

Regras:

- baixo risco: todos os modelos visuais completos abaixo dos seus thresholds
  baixos no fold;
- alto risco: pelo menos um modelo acima do seu threshold alto no fold;
- incerto: demais casos;
- se algum modelo/fold nao tiver threshold baixo seguro, nao ha baixo risco
  naquele fold;
- a comparacao externa de estrategias e exploratoria e nao seleciona a regra.

Invariantes registrados:

- `criterio_definido_antes_avaliacao = true`;
- `usa_resultado_externo_para_selecao = false`;
- `baixo_risco_nao_e_liberacao_automatica = true`.

## 4. Thresholds

Threshold baixo: maior threshold da validacao interna com `fn == 0`,
`tn >= minimo_utilidade` e `threshold_baixo < threshold_alto`, em que
`minimo_utilidade = max(5, ceil(0.05 * suporte_nao_contaminada_validacao))`.

Threshold alto: melhor F1 da classe contaminada na validacao interna,
desempatando por recall, precisao e menor numero de falsos positivos.

Thresholds sem zona de baixo risco:

| fold | grupo_externo | modelo | conjunto_features | status_threshold_baixo |
| --- | --- | --- | --- | --- |
| 1.000 | micro_ondas__controle | knn | principal_normalizado | baixo_risco_suspenso |
| 1.000 | micro_ondas__controle | lda | principal_normalizado | baixo_risco_suspenso |
| 1.000 | micro_ondas__controle | mobilenetv2_recortes | nao_aplicavel | baixo_risco_suspenso |
| 1.000 | micro_ondas__controle | random_forest | principal_normalizado | baixo_risco_suspenso |
| 1.000 | micro_ondas__controle | svm_rbf | principal_normalizado | baixo_risco_suspenso |
| 2.000 | micro_ondas__mw | knn | principal_normalizado | baixo_risco_suspenso |
| 2.000 | micro_ondas__mw | lda | principal_normalizado | baixo_risco_suspenso |
| 2.000 | micro_ondas__mw | mobilenetv2_recortes | nao_aplicavel | baixo_risco_suspenso |
| 2.000 | micro_ondas__mw | random_forest | principal_normalizado | baixo_risco_suspenso |
| 2.000 | micro_ondas__mw | svm_rbf | principal_normalizado | baixo_risco_suspenso |
| 3.000 | piloto__controle | knn | principal_normalizado | baixo_risco_suspenso |
| 3.000 | piloto__controle | lda | principal_normalizado | baixo_risco_suspenso |
| 3.000 | piloto__controle | mobilenetv2_recortes | nao_aplicavel | baixo_risco_suspenso |
| 3.000 | piloto__controle | random_forest | principal_normalizado | baixo_risco_suspenso |
| 3.000 | piloto__controle | svm_rbf | principal_normalizado | baixo_risco_suspenso |
| 4.000 | piloto__fungicida | knn | principal_normalizado | baixo_risco_suspenso |
| 4.000 | piloto__fungicida | lda | principal_normalizado | baixo_risco_suspenso |
| 4.000 | piloto__fungicida | mobilenetv2_recortes | nao_aplicavel | baixo_risco_suspenso |
| 4.000 | piloto__fungicida | random_forest | principal_normalizado | baixo_risco_suspenso |
| 4.000 | piloto__fungicida | svm_rbf | principal_normalizado | baixo_risco_suspenso |

## 5. Resultado oficial

| estrategia_oficial | total | baixo_risco | alto_risco | incerto | contaminadas_baixo_risco | taxa_contaminada_baixo_risco | recall_alto_risco_contaminada | precisao_alto_risco_contaminada | cobertura_decisao | viabilidade_operacional | resultado_cientifico |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| consenso_pre_especificado | 703 | 0 | 703 | 0 | 0 | 0.000 | 1.000 | 0.610 | 1.000 | 0.000 | triagem_nao_viavel_com_base_atual |

Resultado observado nos CSVs consolidados: o consenso oficial classificou
todas as 703 amostras como alto risco; houve 0 amostras incertas e
0 em baixo risco. Zona de baixo risco valida:
nao. O recall de alto risco foi
1.000 e a precisao de alto risco foi
0.610.

Com esse resultado, a regra oficial equivale operacionalmente a uma politica de
cautela total. Ela preserva recall, mas nao apresentou capacidade util de
priorizacao, porque nao criou zona de baixo risco valida nem reduziu o conjunto
encaminhado a alto risco. Viabilidade operacional:
false. Motivo registrado:
`consenso_classificou_todas_amostras_como_alto_risco`. Resultado cientifico: `triagem_nao_viavel_com_base_atual`.

## 6. Micro e macro

Agregacao micro soma todas as amostras antes das metricas. Agregacao macro
resume os grupos externos e e mais sensivel a grupos pequenos.

Resumo micro:

| estrategia | tipo_estrategia | total | baixo_risco | alto_risco | incerto | contaminadas_baixo_risco | recall_alto_risco_contaminada | cobertura_decisao |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| consenso_pre_especificado | consenso_oficial | 703 | 0 | 703 | 0 | 0 | 1.000 | 1.000 |
| individual_knn_principal_normalizado | individual_descritiva | 703 | 0 | 681 | 22 | 0 | 0.984 | 0.969 |
| individual_lda_principal_normalizado | individual_descritiva | 703 | 0 | 496 | 207 | 0 | 0.688 | 0.706 |
| individual_mobilenetv2_recortes_nao_aplicavel | individual_descritiva | 703 | 0 | 631 | 72 | 0 | 0.897 | 0.898 |
| individual_random_forest_principal_normalizado | individual_descritiva | 703 | 0 | 587 | 116 | 0 | 0.853 | 0.835 |
| individual_svm_rbf_principal_normalizado | individual_descritiva | 703 | 0 | 703 | 0 | 0 | 1.000 | 1.000 |

Resumo macro:

| estrategia | tipo_estrategia | grupos | taxa_baixo_risco_media | taxa_alto_risco_media | taxa_incerto_media | recall_alto_risco_contaminada_media | cobertura_decisao_media |
| --- | --- | --- | --- | --- | --- | --- | --- |
| consenso_pre_especificado | consenso_oficial | 12.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| individual_knn_principal_normalizado | individual_descritiva | 12.000 | 0.000 | 0.927 | 0.073 | 0.955 | 0.927 |
| individual_lda_principal_normalizado | individual_descritiva | 12.000 | 0.000 | 0.693 | 0.307 | 0.684 | 0.693 |
| individual_mobilenetv2_recortes_nao_aplicavel | individual_descritiva | 12.000 | 0.000 | 0.842 | 0.158 | 0.837 | 0.842 |
| individual_random_forest_principal_normalizado | individual_descritiva | 12.000 | 0.000 | 0.872 | 0.128 | 0.872 | 0.872 |
| individual_svm_rbf_principal_normalizado | individual_descritiva | 12.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 |

## 7. Casos criticos

Contaminadas em baixo risco:

Sem dados disponiveis.

Todos os casos criticos ficam em `saidas\tabelas\08_triagem\casos_criticos_triagem.csv`.

## 8. Figuras

![Distribuicao](figuras/triagem/distribuicao_decisoes_triagem.png)

![Grupos](figuras/triagem/triagem_por_grupo_consenso.png)

## 9. Manifestos

- `saidas\tabelas\08_triagem\manifesto_thresholds_triagem.json`
- `saidas\tabelas\08_triagem\manifesto_comparacao_triagem.json`

Campos principais:

```json
{
  "thresholds": {
    "protocolo": "triagem_preventiva_crossfit",
    "origem_tabela_integrada": "saidas\\tabelas\\08_triagem\\tabela_integrada_triagem.csv",
    "origem_thresholds_internos": "saidas\\tabelas\\08_triagem\\thresholds_internos_modelos_triagem.csv",
    "threshold_baixo": {
      "criterio": "maior_threshold_com_fn_0_tn_minimo_e_menor_que_threshold_alto",
      "formula_minimo_utilidade": "max(5, ceil(0.05 * suporte_nao_contaminada_validacao))",
      "coluna_minimo_utilidade": "minimo_utilidade_baixo_risco",
      "minimos_utilidade_por_modelo_fold": [
        {
          "fold": 1,
          "grupo_externo": "micro_ondas__controle",
          "modelo": "knn",
          "conjunto_features": "principal_normalizado",
          "suporte_nao_contaminada_validacao": 31,
          "minimo_utilidade_baixo_risco": 5
        },
        {
          "fold": 1,
          "grupo_externo": "micro_ondas__controle",
          "modelo": "lda",
          "conjunto_features": "principal_normalizado",
          "suporte_nao_contaminada_validacao": 31,
          "minimo_utilidade_baixo_risco": 5
        },
        {
          "fold": 1,
          "grupo_externo": "micro_ondas__controle",
          "modelo": "mobilenetv2_recortes",
          "conjunto_features": "nao_aplicavel",
          "suporte_nao_contaminada_validacao": 31,
          "minimo_utilidade_baixo_risco": 5
        },
        {
          "fold": 1,
          "grupo_externo": "micro_ondas__controle",
          "modelo": "random_forest",
          "conjunto_features": "principal_normalizado",
          "suporte_nao_contaminada_validacao": 31,
          "minimo_utilidade_baixo_risco": 5
        },
        {
          "fold": 1,
          "grupo_externo": "micro_ondas__controle",
          "modelo": "svm_rbf",
          "conjunto_features": "principal_normalizado",
          "suporte_nao_contaminada_validacao": 31,
          "minimo_utilidade_baixo_risco": 5
        },
        {
          "fold": 2,
          "grupo_externo": "micro_ondas__mw",
          "modelo": "knn",
          "conjunto_features": "principal_normalizado",
          "suporte_nao_contaminada_validacao": 31,
          "minimo_utilidade_baixo_risco": 5
        },
        {
          "fold": 2,
          "grupo_externo": "micro_ondas__mw",
          "modelo": "lda",
          "conjunto_features": "principal_normalizado",
          "suporte_nao_contaminada_validacao": 31,
          "minimo_utilidade_baixo_risco": 5
        },
        {
          "fold": 2,
          "grupo_externo": "micro_ondas__mw",
          "modelo": "mobilenetv2_recortes",
          "conjunto_features": "nao_aplicavel",
          "suporte_nao_contaminada_validacao": 31,
          "minimo_utilidade_baixo_risco": 5
        },
        {
          "fold": 2,
          "grupo_externo": "micro_ondas__mw",
          "modelo": "random_forest",
          "conjunto_features": "principal_normalizado",
          "suporte_nao_contaminada_validacao": 31,
          "minimo_utilidade_baixo_risco": 5
        },
        {
          "fold": 2,
          "grupo_externo": "micro_ondas__mw",
          "modelo": "svm_rbf",
          "conjunto_features": "principal_normalizado",
          "suporte_nao_contaminada_validacao": 31,
          "minimo_utilidade_baixo_risco": 5
        },
        {
          "fold": 3,
          "grupo_externo": "piloto__controle",
          "modelo": "knn",
          "conjunto_features": "principal_normalizado",
          "suporte_nao_contaminada_validacao": 31,
          "minimo_utilidade_baixo_risco": 5
        },
        {
          "fold": 3,
          "grupo_externo": "piloto__controle",
          "modelo": "lda",
          "conjunto_features": "principal_normalizado",
          "suporte_nao_contaminada_validacao": 31,
          "minimo_utilidade_baixo_risco": 5
        },
        {
          "fold": 3,
          "grupo_externo": "piloto__controle",
          "modelo": "mobilenetv2_recortes",
          "conjunto_features": "
```

## 10. Conclusao

A triagem preventiva foi avaliada sem selecionar regra por desempenho externo.
O consenso pre-especificado permanece como estrategia oficial, e as estrategias
individuais continuam apenas como analises secundarias e descritivas; nenhuma
delas deve ser promovida a oficial depois de olhar a validacao externa.

O resultado observado foi cautela total: todas as 703 amostras em alto risco,
0 incertas e 0 em baixo risco. Com a base atual,
a triagem automatica nao foi considerada operacionalmente viavel.
