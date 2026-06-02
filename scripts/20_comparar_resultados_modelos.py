from pathlib import Path

import pandas as pd


# ============================================================
# SCRIPT 20 - COMPARAR RESULTADOS DOS MODELOS
# ------------------------------------------------------------
# Objetivo:
# - Juntar metricas de baseline, YOLO e recortes em uma tabela
# - Facilitar a escolha do melhor proximo experimento
# - Destacar sensibilidade e especificidade
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"

ARQUIVOS_METRICAS = [
    {
        "modelo": "baseline_resnet18_imagem_inteira",
        "caminho": PASTA_TABELAS / "metricas_baseline_resnet18_teste.csv",
    },
    {
        "modelo": "yolo_caixas_automaticas",
        "caminho": PASTA_TABELAS / "metricas_yolo_teste.csv",
    },
    {
        "modelo": "recortes_resnet18",
        "caminho": PASTA_TABELAS / "metricas_recortes_resnet18_teste.csv",
    },
]

COLUNAS_PRINCIPAIS = [
    "modelo",
    "cenario",
    "threshold",
    "acuracia",
    "precisao_contaminada",
    "recall_contaminada",
    "sensibilidade_contaminada",
    "especificidade_nao_contaminada",
    "f1_contaminada",
    "tn",
    "fp",
    "fn",
    "tp",
    "arquivo_origem",
]

CAMINHO_SAIDA = PASTA_TABELAS / "comparacao_modelos_teste.csv"


def carregar_metricas() -> pd.DataFrame:
    tabelas = []
    ausentes = []

    for item in ARQUIVOS_METRICAS:
        caminho = item["caminho"]
        if not caminho.exists():
            ausentes.append(caminho)
            continue

        df = pd.read_csv(caminho)

        if "modelo" not in df.columns:
            df.insert(0, "modelo", item["modelo"])
        else:
            df["modelo"] = df["modelo"].fillna(item["modelo"])

        df["arquivo_origem"] = str(caminho.relative_to(PASTA_PROJETO))

        if "sensibilidade_contaminada" not in df.columns and "recall_contaminada" in df.columns:
            df["sensibilidade_contaminada"] = df["recall_contaminada"]

        if "especificidade_nao_contaminada" not in df.columns:
            df["especificidade_nao_contaminada"] = df.apply(calcular_especificidade, axis=1)

        tabelas.append(df)

    if ausentes:
        print("Arquivos ainda nao encontrados:")
        for caminho in ausentes:
            print(f"- {caminho}")
        print()

    if not tabelas:
        raise FileNotFoundError("Nenhum arquivo de metricas foi encontrado para comparar.")

    return pd.concat(tabelas, ignore_index=True)


def calcular_especificidade(linha) -> float:
    tn = float(linha.get("tn", 0))
    fp = float(linha.get("fp", 0))
    return tn / max(tn + fp, 1)


def organizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    for coluna in COLUNAS_PRINCIPAIS:
        if coluna not in df.columns:
            df[coluna] = None

    return df[COLUNAS_PRINCIPAIS].sort_values(
        ["recall_contaminada", "f1_contaminada", "especificidade_nao_contaminada"],
        ascending=[False, False, False],
    )


def main():
    print("=" * 60)
    print("COMPARANDO RESULTADOS DOS MODELOS")
    print("=" * 60)

    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)

    df = carregar_metricas()
    df_comparacao = organizar_colunas(df)
    df_comparacao.to_csv(CAMINHO_SAIDA, index=False, encoding="utf-8-sig")

    print()
    print("Comparacao no teste:")
    print(df_comparacao.to_string(index=False))
    print()
    print(f"Arquivo gerado: {CAMINHO_SAIDA}")


if __name__ == "__main__":
    main()
