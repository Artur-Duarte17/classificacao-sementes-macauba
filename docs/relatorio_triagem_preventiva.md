# Relatorio da triagem preventiva

Gerado em: 2026-06-08T15:25:54

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
quantidade minima de nao contaminadas em baixo risco e
`threshold_baixo < threshold_alto`.

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

| estrategia_oficial | total | baixo_risco | alto_risco | incerto | contaminadas_baixo_risco | taxa_contaminada_baixo_risco | recall_alto_risco_contaminada | cobertura_decisao |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| consenso_pre_especificado | 703 | 0 | 703 | 0 | 0 | 0.000 | 1.000 | 1.000 |

Interpretação: se houver contaminadas em baixo risco, isso nao ajusta a regra
pos-hoc, mas impede interpretar baixo risco como liberacao automatica.

## 6. Micro e macro

Agregacao micro soma todas as amostras antes das metricas. Agregacao macro
resume os grupos externos e e mais sensivel a grupos pequenos.

Resumo micro:

| estrategia | tipo_estrategia | total | baixo_risco | alto_risco | incerto | contaminadas_baixo_risco | recall_alto_risco_contaminada | cobertura_decisao |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| consenso_pre_especificado | consenso_oficial | 703 | 0 | 703 | 0 | 0 | 1.000 | 1.000 |
| individual_knn_principal_normalizado | individual_descritiva | 703 | 0 | 681 | 22 | 0 | 0.984 | 0.969 |
| individual_lda_principal_normalizado | individual_descritiva | 703 | 0 | 496 | 207 | 0 | 0.688 | 0.706 |
| individual_mobilenetv2_recortes_nao_aplicavel | individual_descritiva | 703 | 6 | 631 | 66 | 5 | 0.897 | 0.906 |
| individual_random_forest_principal_normalizado | individual_descritiva | 703 | 0 | 587 | 116 | 0 | 0.853 | 0.835 |
| individual_svm_rbf_principal_normalizado | individual_descritiva | 703 | 0 | 703 | 0 | 0 | 1.000 | 1.000 |

Resumo macro:

| estrategia | tipo_estrategia | grupos | taxa_baixo_risco_media | taxa_alto_risco_media | taxa_incerto_media | recall_alto_risco_contaminada_media | cobertura_decisao_media |
| --- | --- | --- | --- | --- | --- | --- | --- |
| consenso_pre_especificado | consenso_oficial | 12.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| individual_knn_principal_normalizado | individual_descritiva | 12.000 | 0.000 | 0.927 | 0.073 | 0.955 | 0.927 |
| individual_lda_principal_normalizado | individual_descritiva | 12.000 | 0.000 | 0.693 | 0.307 | 0.684 | 0.693 |
| individual_mobilenetv2_recortes_nao_aplicavel | individual_descritiva | 12.000 | 0.006 | 0.842 | 0.152 | 0.837 | 0.848 |
| individual_random_forest_principal_normalizado | individual_descritiva | 12.000 | 0.000 | 0.872 | 0.128 | 0.872 | 0.872 |
| individual_svm_rbf_principal_normalizado | individual_descritiva | 12.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 |

## 7. Casos criticos

Contaminadas em baixo risco:

| estrategia | grupo_externo | nome_arquivo | classe_real | decisao_triagem | tipo_caso_critico |
| --- | --- | --- | --- | --- | --- |
| individual_mobilenetv2_recortes_nao_aplicavel | teste_2__t3 | TESTE_2__T3__b1.jpg | contaminada | baixo_risco | contaminada_em_baixo_risco |
| individual_mobilenetv2_recortes_nao_aplicavel | teste_2__t3 | TESTE_2__T3__e5.jpg | contaminada | baixo_risco | contaminada_em_baixo_risco |
| individual_mobilenetv2_recortes_nao_aplicavel | teste_2__t3 | TESTE_2__T3__j3.jpg | contaminada | baixo_risco | contaminada_em_baixo_risco |
| individual_mobilenetv2_recortes_nao_aplicavel | teste_2__t3 | TESTE_2__T3__l4.jpg | contaminada | baixo_risco | contaminada_em_baixo_risco |
| individual_mobilenetv2_recortes_nao_aplicavel | teste_2__t3 | TESTE_2__T3__l5.jpg | contaminada | baixo_risco | contaminada_em_baixo_risco |

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
      "min_nao_contaminadas_baixo_risco": 1,
      "sem_candidato": "nao_existe_zona_de_baixo_risco_modelo_fold"
    },
    "threshold_alto": {
      "criterio": "melhor_f1_desempate_recall_precisao_menor_fp"
    },
    "estrategia_oficial": "consenso_pre_especificado",
    "criterio_definido_antes_avaliacao": true,
    "usa_resultado_externo_para_selecao": false,
    "arquivos_saida": {
      "thresholds": "saidas\\tabelas\\08_triagem\\thresholds_crossfit_por_grupo.csv",
      "predicoes": "saidas\\tabelas\\08_triagem\\predicoes_triagem_crossfit.csv",
      "casos_criticos": "saidas\\tabelas\\08_triagem\\casos_criticos_triagem.csv"
    }
  },
  "comparacao": {
    "protocolo": "triagem_preventiva_crossfit",
    "origem_predicoes": "saidas\\tabelas\\08_triagem\\predicoes_triagem_crossfit.csv",
    "estrategia_oficial": "consenso_pre_especificado",
    "criterio_definido_antes_avaliacao": true,
    "usa_resultado_externo_para_selecao": false,
    "comparacao_exploratoria": "saidas\\tabelas\\08_triagem\\comparacao_scores_triagem.csv",
    "ranking_externo_nao_utilizado_para_selecao": true,
    "arquivos_saida": {
      "metricas_grupo": "saidas\\tabelas\\08_triagem\\metricas_triagem_por_grupo.csv",
      "resumo": "saidas\\tabelas\\08_triagem\\resumo_triagem_micro_macro.csv",
      "comparacao": "saidas\\tabelas\\08_triagem\\comparacao_scores_triagem.csv",
      "recomendado": "saidas\\tabelas\\08_triagem\\score_triagem_recomendado.csv"
    }
  }
}
```

## 10. Conclusao

A triagem preventiva fica cientificamente mais defensavel que a classificacao
automatica direta porque preserva incerteza e nao escolhe regras por desempenho
externo. O consenso pre-especificado e a regra oficial; estrategias individuais
servem apenas como analises secundarias e descritivas.
