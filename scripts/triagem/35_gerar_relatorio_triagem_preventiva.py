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
# - Documentar a estrategia oficial pre-especificada
# - Nao treinar modelos, nao recalibrar thresholds e nao escolher por teste
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


def tabela_markdown(df: pd.DataFrame, colunas: list[str], max_linhas: int = 12) -> str:
    if df.empty:
        return "Sem dados disponiveis."
    dados = df.copy()
    for coluna in colunas:
        if coluna not in dados.columns:
            dados[coluna] = pd.NA
    dados = dados[colunas].head(max_linhas).copy()
    for coluna in dados.columns:
        if pd.api.types.is_numeric_dtype(dados[coluna]):
            if coluna in {
                "total",
                "contaminadas",
                "nao_contaminadas",
                "baixo_risco",
                "alto_risco",
                "incerto",
                "contaminadas_baixo_risco",
                "nao_contaminadas_baixo_risco",
            }:
                dados[coluna] = dados[coluna].map(
                    lambda valor: "NA" if pd.isna(valor) else str(int(round(float(valor))))
                )
            else:
                dados[coluna] = dados[coluna].map(formatar_numero)
        else:
            dados[coluna] = dados[coluna].fillna("NA").astype(str)
    cabecalho = "| " + " | ".join(dados.columns) + " |"
    separador = "| " + " | ".join(["---"] * len(dados.columns)) + " |"
    linhas = ["| " + " | ".join(map(str, linha)) + " |" for linha in dados.to_numpy()]
    return "\n".join([cabecalho, separador, *linhas])


def plotar_distribuicao(resumo: pd.DataFrame) -> None:
    micro = resumo[resumo["agregacao"].astype(str).eq("micro")].copy()
    if micro.empty:
        return
    colunas = ["taxa_baixo_risco", "taxa_alto_risco", "taxa_incerto"]
    dados = micro.set_index("estrategia")[colunas].apply(pd.to_numeric, errors="coerce")
    ax = dados.plot(kind="bar", figsize=(11, 6))
    ax.set_title("Distribuicao das decisoes de triagem por estrategia")
    ax.set_ylabel("Proporcao das amostras")
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
    ax.set_title("Consenso pre-especificado por grupo externo")
    ax.set_ylabel("Proporcao das amostras")
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
    thresholds_sem_baixo = thresholds[
        ~thresholds["baixo_risco_disponivel"].astype(bool)
    ].copy()
    casos_baixo = casos[
        casos.get("tipo_caso_critico", pd.Series(dtype=str)).astype(str)
        == "contaminada_em_baixo_risco"
    ].copy()

    figuras = []
    if FIGURA_DISTRIBUICAO.exists():
        figuras.append(f"![Distribuicao]({caminho_relativo_docs(FIGURA_DISTRIBUICAO)})")
    if FIGURA_GRUPOS.exists():
        figuras.append(f"![Grupos]({caminho_relativo_docs(FIGURA_GRUPOS)})")
    figuras_md = "\n\n".join(figuras) if figuras else "Figuras nao geradas."

    relatorio = f"""
# Relatorio da triagem preventiva

Gerado em: {datetime.now().isoformat(timespec='seconds')}

## 1. Objetivo

A triagem preventiva transforma os resultados de classificacao em uma regra
operacional conservadora. Ela nao substitui a avaliacao manual: o objetivo e
priorizar alto risco, manter incerteza quando o sinal nao e suficiente e evitar
tratar baixo risco como liberacao automatica.

## 2. Entradas

- `{caminho_relativo(CAMINHO_TABELA_INTEGRADA)}`
- `{caminho_relativo(CAMINHO_SCORES)}`
- `{caminho_relativo(CAMINHO_THRESHOLDS)}`
- `{caminho_relativo(CAMINHO_PREDICOES)}`
- `{caminho_relativo(CAMINHO_METRICAS_GRUPO)}`
- `{caminho_relativo(CAMINHO_RESUMO)}`
- `{caminho_relativo(CAMINHO_RECOMENDADO)}`

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

{tabela_markdown(thresholds_sem_baixo, ['fold', 'grupo_externo', 'modelo', 'conjunto_features', 'status_threshold_baixo'], 20)}

## 5. Resultado oficial

{tabela_markdown(recomendado, ['estrategia_oficial', 'total', 'baixo_risco', 'alto_risco', 'incerto', 'contaminadas_baixo_risco', 'taxa_contaminada_baixo_risco', 'recall_alto_risco_contaminada', 'cobertura_decisao'], 1)}

Interpretação: se houver contaminadas em baixo risco, isso nao ajusta a regra
pos-hoc, mas impede interpretar baixo risco como liberacao automatica.

## 6. Micro e macro

Agregacao micro soma todas as amostras antes das metricas. Agregacao macro
resume os grupos externos e e mais sensivel a grupos pequenos.

Resumo micro:

{tabela_markdown(micro, ['estrategia', 'tipo_estrategia', 'total', 'baixo_risco', 'alto_risco', 'incerto', 'contaminadas_baixo_risco', 'recall_alto_risco_contaminada', 'cobertura_decisao'], 12)}

Resumo macro:

{tabela_markdown(macro, ['estrategia', 'tipo_estrategia', 'grupos', 'taxa_baixo_risco_media', 'taxa_alto_risco_media', 'taxa_incerto_media', 'recall_alto_risco_contaminada_media', 'cobertura_decisao_media'], 12)}

## 7. Casos criticos

Contaminadas em baixo risco:

{tabela_markdown(casos_baixo, ['estrategia', 'grupo_externo', 'nome_arquivo', 'classe_real', 'decisao_triagem', 'tipo_caso_critico'], 20)}

Todos os casos criticos ficam em `{caminho_relativo(CAMINHO_CASOS_CRITICOS)}`.

## 8. Figuras

{figuras_md}

## 9. Manifestos

- `{caminho_relativo(CAMINHO_MANIFESTO_THRESHOLDS)}`
- `{caminho_relativo(CAMINHO_MANIFESTO_COMPARACAO)}`

Campos principais:

```json
{json.dumps({
    'thresholds': manifesto_thresholds,
    'comparacao': manifesto_comparacao,
}, indent=2, ensure_ascii=False)[:4000]}
```

## 10. Conclusao

A triagem preventiva fica cientificamente mais defensavel que a classificacao
automatica direta porque preserva incerteza e nao escolhe regras por desempenho
externo. O consenso pre-especificado e a regra oficial; estrategias individuais
servem apenas como analises secundarias e descritivas.
"""
    return textwrap.dedent(relatorio).strip() + "\n"


def main() -> None:
    print("=" * 70)
    print("GERANDO RELATORIO DA TRIAGEM PREVENTIVA")
    print("=" * 70)
    print("Este script nao treina modelos e nao recalibra thresholds.")

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

    print(f"Relatorio: {CAMINHO_RELATORIO}")
    print(f"Figuras: {PASTA_FIGURAS_DOCS}")


if __name__ == "__main__":
    main()
