from pathlib import Path
from datetime import datetime
import json
import textwrap

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# SCRIPT 35 - GERAR RELATORIO DA TRIAGEM PREVENTIVA
# ------------------------------------------------------------
# Objetivo:
# - Consolidar a triagem crossfit ja gerada
# - Documentar a estratégia oficial pré-especificada
# - Não treinar modelos, não recalibrar thresholds e não escolher por teste
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_DOCS = PASTA_PROJETO / "docs"
PASTA_FIGURAS_DOCS = PASTA_DOCS / "figuras" / "triagem"
PASTA_TRIAGEM = PASTA_PROJETO / "saidas" / "tabelas" / "08_triagem"

CAMINHO_TABELA_INTEGRADA = PASTA_TRIAGEM / "tabela_integrada_triagem.csv"
CAMINHO_SCORES = PASTA_TRIAGEM / "scores_candidatos_triagem.csv"
CAMINHO_THRESHOLDS = PASTA_TRIAGEM / "thresholds_crossfit_por_grupo.csv"
CAMINHO_PREDICOES = PASTA_TRIAGEM / "predicoes_triagem_crossfit.csv"
CAMINHO_METRICAS_GRUPO = PASTA_TRIAGEM / "metricas_triagem_por_grupo.csv"
CAMINHO_RESUMO = PASTA_TRIAGEM / "resumo_triagem_micro_macro.csv"
CAMINHO_COMPARACAO = PASTA_TRIAGEM / "comparacao_scores_triagem.csv"
CAMINHO_RECOMENDADO = PASTA_TRIAGEM / "score_triagem_recomendado.csv"
CAMINHO_CASOS_CRITICOS = PASTA_TRIAGEM / "casos_criticos_triagem.csv"
CAMINHO_MANIFESTO_THRESHOLDS = PASTA_TRIAGEM / "manifesto_thresholds_triagem.json"
CAMINHO_MANIFESTO_COMPARACAO = PASTA_TRIAGEM / "manifesto_comparacao_triagem.json"

CAMINHO_RELATORIO = PASTA_DOCS / "relatorio_triagem_preventiva.md"
FIGURA_DISTRIBUICAO = PASTA_FIGURAS_DOCS / "distribuicao_decisoes_triagem.png"
FIGURA_GRUPOS = PASTA_FIGURAS_DOCS / "triagem_por_grupo_consenso.png"

COLUNAS_INTEIRAS = {
    "fold",
    "grupos",
    "total",
    "contaminadas",
    "nao_contaminadas",
    "baixo_risco",
    "alto_risco",
    "incerto",
    "contaminadas_baixo_risco",
    "nao_contaminadas_baixo_risco",
    "contaminadas_alto_risco",
    "nao_contaminadas_alto_risco",
    "contaminadas_incerto",
    "nao_contaminadas_incerto",
}

NOMES_ESTRATEGIAS = {
    "consenso_pre_especificado": "Consenso oficial",
    "individual_knn_principal_normalizado": "k-NN",
    "individual_lda_principal_normalizado": "LDA",
    "individual_mobilenetv2_recortes_nao_aplicavel": "MobileNetV2",
    "individual_random_forest_principal_normalizado": "Random Forest",
    "individual_svm_rbf_principal_normalizado": "SVM RBF",
}

ROTULOS_COLUNAS = {
    "campo": "campo",
    "valor": "valor",
    "fold": "fold",
    "grupo_externo": "grupo externo",
    "modelo": "modelo",
    "conjunto_features": "conjunto de features",
    "status_threshold_baixo": "status threshold baixo",
    "estrategia": "estratégia",
    "estrategia_oficial": "estratégia oficial",
    "tipo_estrategia": "tipo de estratégia",
    "total": "total",
    "baixo_risco": "baixo risco",
    "alto_risco": "alto risco",
    "incerto": "incerto",
    "contaminadas_baixo_risco": "contaminadas em baixo risco",
    "taxa_contaminada_baixo_risco": "taxa contaminada em baixo risco",
    "recall_alto_risco_contaminada": "recall alto risco",
    "precisao_alto_risco_contaminada": "precisão alto risco",
    "cobertura_decisao": "cobertura da decisão",
    "viabilidade_operacional": "viabilidade operacional",
    "resultado_cientifico": "resultado científico",
    "grupos": "grupos",
    "taxa_baixo_risco_media": "taxa baixo risco média",
    "taxa_alto_risco_media": "taxa alto risco média",
    "taxa_incerto_media": "taxa incerto média",
    "recall_alto_risco_contaminada_media": "recall alto risco médio",
    "cobertura_decisao_media": "cobertura da decisão média",
    "nome_arquivo": "arquivo",
    "classe_real": "classe real",
    "decisao_triagem": "decisão da triagem",
    "tipo_caso_critico": "tipo de caso crítico",
}


def ler_csv_obrigatorio(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo obrigatorio ausente: {caminho}")
    df = pd.read_csv(caminho)
    if df.empty:
        raise ValueError(f"Arquivo obrigatorio vazio: {caminho}")
    return df


def ler_csv_opcional(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(caminho)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def ler_json_opcional(caminho: Path) -> dict:
    if not caminho.exists():
        return {}
    return json.loads(caminho.read_text(encoding="utf-8"))


def caminho_relativo_docs(caminho: Path) -> str:
    return str(caminho.relative_to(PASTA_DOCS)).replace("\\", "/")


def caminho_relativo(caminho: Path) -> str:
    return str(caminho.relative_to(PASTA_PROJETO))


def formatar_numero(valor, casas: int = 3) -> str:
    if pd.isna(valor):
        return "NA"
    try:
        return f"{float(valor):.{casas}f}"
    except (TypeError, ValueError):
        return str(valor)


def formatar_numero_texto(valor, casas: int = 3) -> str:
    return formatar_numero(valor, casas).replace(".", ",")


def formatar_inteiro(valor) -> str:
    if pd.isna(valor):
        return "NA"
    try:
        return str(int(round(float(valor))))
    except (TypeError, ValueError):
        return str(valor)


def nome_estrategia_legivel(estrategia: str) -> str:
    estrategia = str(estrategia)
    if estrategia in NOMES_ESTRATEGIAS:
        return NOMES_ESTRATEGIAS[estrategia]
    if "mobilenetv2" in estrategia:
        return "MobileNetV2"
    if "random_forest" in estrategia:
        return "Random Forest"
    if "svm_rbf" in estrategia:
        return "SVM RBF"
    if "knn" in estrategia:
        return "k-NN"
    if "lda" in estrategia:
        return "LDA"
    return estrategia


def normalizar_bool(serie: pd.Series) -> pd.Series:
    if serie.dtype == bool:
        return serie
    return serie.astype(str).str.lower().isin(["true", "1", "sim"])


def obter_valor(linha: pd.Series, coluna: str, padrao=None):
    return linha[coluna] if coluna in linha.index else padrao


def valor_bool(valor) -> bool:
    if pd.isna(valor):
        return False
    if isinstance(valor, bool):
        return valor
    return str(valor).lower() in {"true", "1", "sim"}


def tabela_manifestos(
    manifesto_thresholds: dict,
    manifesto_comparacao: dict,
    recomendado: pd.Series,
) -> str:
    threshold_baixo = manifesto_thresholds.get("threshold_baixo", {})
    registros = [
        {
            "campo": "protocolo",
            "valor": (
                manifesto_comparacao.get("protocolo")
                or manifesto_thresholds.get("protocolo")
                or "triagem_preventiva_crossfit"
            ),
        },
        {
            "campo": "abreviacao_crossfit",
            "valor": (
                "nome técnico interno dos arquivos; corresponde a validação externa "
                "leave-one-experimento-tratamento-out com calibração interna por grupo, "
                "não a cross-fitting estatístico clássico"
            ),
        },
        {
            "campo": "estrategia_oficial",
            "valor": (
                manifesto_comparacao.get("estrategia_oficial")
                or manifesto_thresholds.get("estrategia_oficial")
                or "consenso_pre_especificado"
            ),
        },
        {
            "campo": "criterio_definido_antes_avaliacao",
            "valor": str(
                manifesto_comparacao.get("criterio_definido_antes_avaliacao", True)
            ).lower(),
        },
        {
            "campo": "usa_resultado_externo_para_selecao",
            "valor": str(
                manifesto_comparacao.get("usa_resultado_externo_para_selecao", False)
            ).lower(),
        },
        {
            "campo": "formula_minimo_utilidade",
            "valor": threshold_baixo.get(
                "formula_minimo_utilidade",
                "max(5, ceil(0.05 * suporte_nao_contaminada_validacao))",
            ),
        },
        {
            "campo": "resultado_cientifico",
            "valor": obter_valor(
                recomendado,
                "resultado_cientifico",
                "triagem_nao_viavel_com_base_atual",
            ),
        },
    ]
    return tabela_markdown(pd.DataFrame(registros), ["campo", "valor"], len(registros))


def tabela_markdown(df: pd.DataFrame, colunas: list[str], max_linhas: int = 12) -> str:
    if df.empty:
        return "Sem dados disponíveis."
    dados = df.copy()
    for coluna in colunas:
        if coluna not in dados.columns:
            dados[coluna] = pd.NA
    dados = dados[colunas].head(max_linhas).copy()
    for coluna in dados.columns:
        if pd.api.types.is_numeric_dtype(dados[coluna]):
            if coluna in COLUNAS_INTEIRAS:
                dados[coluna] = dados[coluna].map(formatar_inteiro)
            else:
                dados[coluna] = dados[coluna].map(formatar_numero)
        else:
            dados[coluna] = dados[coluna].fillna("NA").astype(str)
        if coluna == "estrategia":
            dados[coluna] = dados[coluna].map(nome_estrategia_legivel)
    cabecalhos = [ROTULOS_COLUNAS.get(coluna, coluna) for coluna in dados.columns]
    cabecalho = "| " + " | ".join(cabecalhos) + " |"
    separador = "| " + " | ".join(["---"] * len(dados.columns)) + " |"
    linhas = ["| " + " | ".join(map(str, linha)) + " |" for linha in dados.to_numpy()]
    return "\n".join([cabecalho, separador, *linhas])


def plotar_distribuicao(resumo: pd.DataFrame) -> None:
    micro = resumo[resumo["agregacao"].astype(str).eq("micro")].copy()
    if micro.empty:
        return
    colunas = ["taxa_baixo_risco", "taxa_alto_risco", "taxa_incerto"]
    micro["estrategia_legivel"] = micro["estrategia"].map(nome_estrategia_legivel)
    dados = micro.set_index("estrategia_legivel")[colunas].apply(
        pd.to_numeric,
        errors="coerce",
    )
    ax = dados.plot(kind="bar", figsize=(11, 6))
    ax.set_title("Distribuição das decisões de triagem por estratégia")
    ax.set_ylabel("Proporção das amostras")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(["Baixo risco", "Alto risco", "Incerto"], ncol=3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    PASTA_FIGURAS_DOCS.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURA_DISTRIBUICAO, dpi=160)
    plt.close()


def plotar_grupos(metricas_grupo: pd.DataFrame) -> None:
    dados = metricas_grupo[
        metricas_grupo["estrategia"].astype(str).eq("consenso_pre_especificado")
    ].copy()
    if dados.empty:
        return
    dados = dados.sort_values("grupo_externo")
    ax = dados.plot(
        x="grupo_externo",
        y=["taxa_baixo_risco", "taxa_alto_risco", "taxa_incerto"],
        kind="bar",
        figsize=(12, 6),
    )
    ax.set_title("Consenso oficial por grupo externo")
    ax.set_ylabel("Proporção das amostras")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(["Baixo risco", "Alto risco", "Incerto"], ncol=3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    PASTA_FIGURAS_DOCS.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURA_GRUPOS, dpi=160)
    plt.close()


def gerar_relatorio(
    resumo: pd.DataFrame,
    metricas_grupo: pd.DataFrame,
    thresholds: pd.DataFrame,
    recomendado: pd.DataFrame,
    casos: pd.DataFrame,
    manifesto_thresholds: dict,
    manifesto_comparacao: dict,
) -> str:
    oficial = recomendado.iloc[0]
    micro = resumo[resumo["agregacao"].astype(str).eq("micro")].copy()
    macro = resumo[resumo["agregacao"].astype(str).eq("macro")].copy()
    baixo_disponivel = normalizar_bool(thresholds["baixo_risco_disponivel"])
    thresholds_sem_baixo = thresholds[~baixo_disponivel].copy()
    existe_zona_baixo = bool(baixo_disponivel.any())
    casos_baixo = casos[
        casos.get("tipo_caso_critico", pd.Series(dtype=str)).astype(str)
        == "contaminada_em_baixo_risco"
    ].copy()
    total_oficial = int(round(float(obter_valor(oficial, "total", 0))))
    alto_oficial = int(round(float(obter_valor(oficial, "alto_risco", 0))))
    baixo_oficial = int(round(float(obter_valor(oficial, "baixo_risco", 0))))
    incerto_oficial = int(round(float(obter_valor(oficial, "incerto", 0))))
    recall_alto = float(obter_valor(oficial, "recall_alto_risco_contaminada", 0.0))
    precisao_alto = float(obter_valor(oficial, "precisao_alto_risco_contaminada", 0.0))
    todas_alto = alto_oficial == total_oficial and total_oficial > 0
    texto_todas_alto = (
        f"todas as {total_oficial} amostras"
        if todas_alto
        else f"{alto_oficial} de {total_oficial} amostras"
    )
    viabilidade_operacional = valor_bool(
        obter_valor(oficial, "viabilidade_operacional", False)
    )
    motivo_inviabilidade = obter_valor(
        oficial,
        "motivo_inviabilidade",
        "nao_definido",
    )
    resultado_cientifico = obter_valor(
        oficial,
        "resultado_cientifico",
        "nao_definido",
    )

    figuras = []
    if FIGURA_DISTRIBUICAO.exists():
        figuras.append(f"![Distribuição]({caminho_relativo_docs(FIGURA_DISTRIBUICAO)})")
    if FIGURA_GRUPOS.exists():
        figuras.append(f"![Grupos]({caminho_relativo_docs(FIGURA_GRUPOS)})")
    figuras_md = "\n\n".join(figuras) if figuras else "Figuras não geradas."
    manifestos_md = tabela_manifestos(
        manifesto_thresholds,
        manifesto_comparacao,
        oficial,
    )

    relatorio = f"""
# Relatório da triagem preventiva

Gerado em: {datetime.now().isoformat(timespec='seconds')}

## 1. Objetivo

A triagem preventiva transforma os resultados de classificação em uma regra
operacional conservadora. Ela não substitui a avaliação manual: o objetivo é
priorizar alto risco, manter incerteza quando o sinal não é suficiente e evitar
tratar baixo risco como liberação automática.

## 2. Entradas

- `{caminho_relativo(CAMINHO_TABELA_INTEGRADA)}`
- `{caminho_relativo(CAMINHO_SCORES)}`
- `{caminho_relativo(CAMINHO_THRESHOLDS)}`
- `{caminho_relativo(CAMINHO_PREDICOES)}`
- `{caminho_relativo(CAMINHO_METRICAS_GRUPO)}`
- `{caminho_relativo(CAMINHO_RESUMO)}`
- `{caminho_relativo(CAMINHO_RECOMENDADO)}`

## 3. Estratégia oficial

A estratégia oficial foi definida antes da avaliação externa:
`consenso_pre_especificado`.

Regras:

- baixo risco: todos os modelos visuais completos abaixo dos seus thresholds
  baixos no fold;
- alto risco: pelo menos um modelo acima do seu threshold alto no fold;
- incerto: demais casos;
- se algum modelo/fold não tiver threshold baixo seguro, não há baixo risco
  naquele fold;
- a comparação externa de estratégias é exploratória e não seleciona a regra.

O termo `crossfit` permanece apenas nos nomes técnicos dos arquivos. Neste
relatório, ele significa validação externa
leave-one-experimento-tratamento-out com calibração interna por grupo; não deve
ser interpretado como cross-fitting estatístico clássico.

Invariantes registrados:

- `criterio_definido_antes_avaliacao = true`;
- `usa_resultado_externo_para_selecao = false`;
- `baixo_risco_nao_e_liberacao_automatica = true`.

## 4. Thresholds

Threshold baixo: maior threshold da validação interna com `fn == 0`,
`tn >= minimo_utilidade` e `threshold_baixo < threshold_alto`, em que
`minimo_utilidade = max(5, ceil(0.05 * suporte_nao_contaminada_validacao))`.

Threshold alto: melhor F1 da classe contaminada na validação interna,
desempatando por recall, precisão e menor número de falsos positivos.

Thresholds sem zona de baixo risco:

{tabela_markdown(thresholds_sem_baixo, ['fold', 'grupo_externo', 'modelo', 'conjunto_features', 'status_threshold_baixo'], 20)}

## 5. Resultado oficial

{tabela_markdown(recomendado, ['estrategia_oficial', 'total', 'baixo_risco', 'alto_risco', 'incerto', 'contaminadas_baixo_risco', 'taxa_contaminada_baixo_risco', 'recall_alto_risco_contaminada', 'precisao_alto_risco_contaminada', 'cobertura_decisao', 'viabilidade_operacional', 'resultado_cientifico'], 1)}

Resultado observado nos CSVs consolidados: o consenso oficial classificou
{texto_todas_alto} como alto risco; houve {incerto_oficial} amostras incertas e
{baixo_oficial} em baixo risco. Zona de baixo risco valida:
{"sim" if existe_zona_baixo else "não"}. O recall de alto risco foi
{formatar_numero_texto(recall_alto)} e a precisão de alto risco foi
aproximadamente {formatar_numero_texto(precisao_alto)}.

Com esse resultado, a regra oficial equivale operacionalmente a uma política de
cautela total. Ela preserva recall, mas não apresentou capacidade útil de
priorização, porque não criou zona de baixo risco válida nem reduziu o conjunto
encaminhado a alto risco. Viabilidade operacional:
{str(viabilidade_operacional).lower()}. Motivo registrado:
`{motivo_inviabilidade}`. Resultado científico: `{resultado_cientifico}`.

## 6. Micro e macro

Agregação micro soma todas as amostras antes das métricas. Agregação macro
resume os grupos externos e é mais sensível a grupos pequenos.

Resumo micro:

{tabela_markdown(micro, ['estrategia', 'tipo_estrategia', 'total', 'baixo_risco', 'alto_risco', 'incerto', 'contaminadas_baixo_risco', 'recall_alto_risco_contaminada', 'cobertura_decisao'], 12)}

Resumo macro:

{tabela_markdown(macro, ['estrategia', 'tipo_estrategia', 'grupos', 'taxa_baixo_risco_media', 'taxa_alto_risco_media', 'taxa_incerto_media', 'recall_alto_risco_contaminada_media', 'cobertura_decisao_media'], 12)}

## 7. Casos críticos

Contaminadas em baixo risco:

{tabela_markdown(casos_baixo, ['estrategia', 'grupo_externo', 'nome_arquivo', 'classe_real', 'decisao_triagem', 'tipo_caso_critico'], 20)}

Todos os casos críticos ficam em `{caminho_relativo(CAMINHO_CASOS_CRITICOS)}`.

## 8. Figuras

{figuras_md}

## 9. Manifestos

- `{caminho_relativo(CAMINHO_MANIFESTO_THRESHOLDS)}`
- `{caminho_relativo(CAMINHO_MANIFESTO_COMPARACAO)}`

Campos principais:

{manifestos_md}

## 10. Conclusão

A triagem preventiva foi avaliada sem selecionar regra por desempenho externo.
O consenso pré-especificado permanece como estratégia oficial, e as estratégias
individuais continuam apenas como análises secundárias e descritivas; nenhuma
delas deve ser promovida a oficial depois de olhar a validação externa.

O resultado observado foi cautela total: {texto_todas_alto} em alto risco,
{incerto_oficial} incertas e {baixo_oficial} em baixo risco. Com a base atual,
a triagem automática não foi considerada operacionalmente viável.
"""
    return textwrap.dedent(relatorio).strip() + "\n"


def main() -> None:
    print("=" * 70)
    print("GERANDO RELATÓRIO DA TRIAGEM PREVENTIVA")
    print("=" * 70)
    print("Este script não treina modelos e não recalibra thresholds.")

    resumo = ler_csv_obrigatorio(CAMINHO_RESUMO)
    metricas_grupo = ler_csv_obrigatorio(CAMINHO_METRICAS_GRUPO)
    thresholds = ler_csv_obrigatorio(CAMINHO_THRESHOLDS)
    recomendado = ler_csv_obrigatorio(CAMINHO_RECOMENDADO)
    casos = ler_csv_opcional(CAMINHO_CASOS_CRITICOS)
    manifesto_thresholds = ler_json_opcional(CAMINHO_MANIFESTO_THRESHOLDS)
    manifesto_comparacao = ler_json_opcional(CAMINHO_MANIFESTO_COMPARACAO)

    plotar_distribuicao(resumo)
    plotar_grupos(metricas_grupo)
    relatorio = gerar_relatorio(
        resumo,
        metricas_grupo,
        thresholds,
        recomendado,
        casos,
        manifesto_thresholds,
        manifesto_comparacao,
    )
    CAMINHO_RELATORIO.write_text(relatorio, encoding="utf-8")

    print(f"Relatório: {CAMINHO_RELATORIO}")
    print(f"Figuras: {PASTA_FIGURAS_DOCS}")


if __name__ == "__main__":
    main()
