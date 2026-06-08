from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# SCRIPT 27 - COMPARAR CLASSIFICACAO FINAL
# ------------------------------------------------------------
# Objetivo:
# - Consolidar metricas finais de classificacao no teste
# - Comparar modelos visuais, modelos classicos, MobileNetV2 e metadados
# - Incluir controles para interpretar recall alto com baixa especificidade
# - Falhar claramente se algum resultado obrigatorio estiver ausente
#
# Este script nao treina modelos e nao altera thresholds/splits/resultados.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_MODELOS = PASTA_TABELAS / "06_modelos"
PASTA_CLASSIFICACAO_FINAL = PASTA_TABELAS / "07_classificacao_final"

CAMINHO_COMPARACAO_FINAL = (
    PASTA_CLASSIFICACAO_FINAL / "comparacao_final_classificacao.csv"
)
CAMINHO_RANKING_EQUILIBRADO = (
    PASTA_CLASSIFICACAO_FINAL / "ranking_equilibrado_classificacao.csv"
)
CAMINHO_RANKING_RECALL = (
    PASTA_CLASSIFICACAO_FINAL / "ranking_prioridade_recall_classificacao.csv"
)
CAMINHO_RESUMO = PASTA_CLASSIFICACAO_FINAL / "resumo_comparacao_classificacao.txt"

TOTAL_TESTE_ESPERADO = 106
SUPORTE_CONTAMINADA_ESPERADO = 65
SUPORTE_NAO_CONTAMINADA_ESPERADO = 41
EPS = 1e-12

CONJUNTO_NAO_APLICAVEL = "nao_aplicavel"
CONJUNTO_PRINCIPAL = "principal_normalizado"
CONJUNTO_SENSIBILIDADE = "sensibilidade_todos_atributos"

ARQUIVOS_METRICAS = [
    {
        "chave": "baseline",
        "modelo_padrao": "baseline_resnet18_imagem_inteira",
        "familia_modelo": "cnn_resnet18",
        "tipo_entrada": "imagem_inteira",
        "usa_pixels": True,
        "usa_recorte": False,
        "usa_atributos_visuais": False,
        "usa_metadados": False,
        "resultado_oficial": True,
        "papel_experimento": "modelo_visual",
        "caminho": PASTA_MODELOS / "baseline" / "metricas_baseline_resnet18_teste.csv",
    },
    {
        "chave": "yolo",
        "modelo_padrao": "yolo_caixas_automaticas",
        "familia_modelo": "yolo",
        "tipo_entrada": "caixas_yolo",
        "usa_pixels": True,
        "usa_recorte": True,
        "usa_atributos_visuais": False,
        "usa_metadados": False,
        "resultado_oficial": True,
        "papel_experimento": "modelo_visual",
        "caminho": PASTA_MODELOS / "yolo" / "metricas_yolo_teste.csv",
    },
    {
        "chave": "recortes_resnet18",
        "modelo_padrao": "recortes_resnet18",
        "familia_modelo": "cnn_resnet18",
        "tipo_entrada": "recorte",
        "usa_pixels": True,
        "usa_recorte": True,
        "usa_atributos_visuais": False,
        "usa_metadados": False,
        "resultado_oficial": True,
        "papel_experimento": "modelo_visual",
        "caminho": PASTA_MODELOS / "recortes" / "metricas_recortes_resnet18_teste.csv",
    },
    {
        "chave": "classicos",
        "modelo_padrao": "modelo_classico_recortes",
        "familia_modelo": "modelo_classico",
        "tipo_entrada": "atributos_visuais_recortes",
        "usa_pixels": False,
        "usa_recorte": True,
        "usa_atributos_visuais": True,
        "usa_metadados": False,
        "resultado_oficial": True,
        "papel_experimento": "modelo_visual_classico",
        "caminho": PASTA_MODELOS / "classicos" / "metricas_classicos_teste.csv",
    },
    {
        "chave": "mobilenetv2",
        "modelo_padrao": "mobilenetv2_recortes",
        "familia_modelo": "cnn_mobilenetv2",
        "tipo_entrada": "recorte",
        "usa_pixels": True,
        "usa_recorte": True,
        "usa_atributos_visuais": False,
        "usa_metadados": False,
        "resultado_oficial": True,
        "papel_experimento": "modelo_visual",
        "caminho": PASTA_MODELOS
        / "mobilenetv2"
        / "metricas_mobilenetv2_recortes_teste.csv",
    },
    {
        "chave": "metadados",
        "modelo_padrao": "metadados_taxas_suavizadas",
        "familia_modelo": "baseline_metadados",
        "tipo_entrada": "metadados",
        "usa_pixels": False,
        "usa_recorte": False,
        "usa_atributos_visuais": False,
        "usa_metadados": True,
        "resultado_oficial": False,
        "papel_experimento": "diagnostico_vies",
        "caminho": PASTA_MODELOS / "metadados" / "metricas_metadados_teste.csv",
    },
]

COLUNAS_CONFUSAO = ["tn", "fp", "fn", "tp"]
COLUNAS_METRICAS_TAXA = [
    "acuracia",
    "precisao_contaminada",
    "recall_contaminada",
    "sensibilidade_contaminada",
    "especificidade_nao_contaminada",
    "f1_contaminada",
    "balanced_accuracy",
    "taxa_predita_contaminada",
]
COLUNAS_METRICAS_LIMITADAS = COLUNAS_METRICAS_TAXA + ["youden_j", "mcc"]
COLUNAS_SAIDA = [
    "modelo",
    "familia_modelo",
    "tipo_entrada",
    "cenario",
    "threshold",
    "usa_pixels",
    "usa_recorte",
    "usa_atributos_visuais",
    "usa_metadados",
    "conjunto_features",
    "resultado_oficial",
    "papel_experimento",
    "arquivo_origem",
    "acuracia",
    "precisao_contaminada",
    "recall_contaminada",
    "sensibilidade_contaminada",
    "especificidade_nao_contaminada",
    "f1_contaminada",
    "balanced_accuracy",
    "youden_j",
    "mcc",
    "taxa_predita_contaminada",
    "total_teste",
    "suporte_contaminada",
    "suporte_nao_contaminada",
    "tn",
    "fp",
    "fn",
    "tp",
]


def caminho_relativo(caminho: Path) -> str:
    return str(caminho.relative_to(PASTA_PROJETO))


def ler_metricas_obrigatorio(item: dict) -> pd.DataFrame:
    caminho = item["caminho"]
    if not caminho.exists():
        raise FileNotFoundError(
            "Arquivo obrigatorio de metricas ausente para a comparacao final: "
            f"{caminho}\n"
            "Gere esse resultado antes de rodar o script 27. "
            "O script 27 nao treina nem recalcula modelos."
        )

    df = pd.read_csv(caminho)
    if df.empty:
        raise ValueError(f"Arquivo de metricas vazio: {caminho}")

    if "modelo" not in df.columns:
        df.insert(0, "modelo", item["modelo_padrao"])
    else:
        df["modelo"] = (
            df["modelo"]
            .fillna(item["modelo_padrao"])
            .replace("", item["modelo_padrao"])
        )

    if "cenario" not in df.columns:
        raise ValueError(f"Coluna obrigatoria ausente em {caminho}: cenario")

    if "conjunto_features" not in df.columns:
        df["conjunto_features"] = CONJUNTO_NAO_APLICAVEL
    df["conjunto_features"] = (
        df["conjunto_features"]
        .fillna(CONJUNTO_NAO_APLICAVEL)
        .replace("", CONJUNTO_NAO_APLICAVEL)
    )

    for coluna in COLUNAS_CONFUSAO:
        if coluna not in df.columns:
            raise ValueError(f"Coluna obrigatoria ausente em {caminho}: {coluna}")
        df[coluna] = pd.to_numeric(df[coluna], errors="raise").astype(int)

    if "threshold" not in df.columns:
        df["threshold"] = np.nan

    df["arquivo_origem"] = caminho_relativo(caminho)
    return df


def calcular_metricas_confusao(tn: int, fp: int, fn: int, tp: int) -> dict:
    total = tn + fp + fn + tp
    suporte_contaminada = tp + fn
    suporte_nao_contaminada = tn + fp

    precisao = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / suporte_contaminada if suporte_contaminada else 0.0
    especificidade = tn / suporte_nao_contaminada if suporte_nao_contaminada else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) else 0.0
    acuracia = (tp + tn) / total if total else 0.0
    balanced_accuracy = (recall + especificidade) / 2
    youden_j = recall + especificidade - 1
    taxa_predita_contaminada = (tp + fp) / total if total else 0.0

    denominador_mcc = math.sqrt(
        max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0)
    )
    mcc = ((tp * tn) - (fp * fn)) / denominador_mcc if denominador_mcc else 0.0

    return {
        "acuracia": float(acuracia),
        "precisao_contaminada": float(precisao),
        "recall_contaminada": float(recall),
        "sensibilidade_contaminada": float(recall),
        "especificidade_nao_contaminada": float(especificidade),
        "f1_contaminada": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "youden_j": float(youden_j),
        "mcc": float(mcc),
        "taxa_predita_contaminada": float(taxa_predita_contaminada),
        "total_teste": int(total),
        "suporte_contaminada": int(suporte_contaminada),
        "suporte_nao_contaminada": int(suporte_nao_contaminada),
    }


def familia_classico(modelo: str) -> str:
    if modelo == "random_forest":
        return "random_forest"
    if modelo == "svm_rbf":
        return "svm_rbf"
    if modelo == "knn":
        return "knn"
    if modelo == "lda":
        return "lda"
    return "modelo_classico"


def enriquecer_metricas(df: pd.DataFrame, item: dict) -> pd.DataFrame:
    df = df.copy()

    for coluna in [
        "familia_modelo",
        "tipo_entrada",
        "usa_pixels",
        "usa_recorte",
        "usa_atributos_visuais",
        "usa_metadados",
        "resultado_oficial",
        "papel_experimento",
    ]:
        df[coluna] = item[coluna]

    if item["chave"] == "classicos":
        df["familia_modelo"] = df["modelo"].astype(str).map(familia_classico)
        df["resultado_oficial"] = df["conjunto_features"].eq(CONJUNTO_PRINCIPAL)
        df["papel_experimento"] = "modelo_visual_classico"

        mascara_rf_sensibilidade = (
            df["modelo"].astype(str).eq("random_forest")
            & df["conjunto_features"].eq(CONJUNTO_SENSIBILIDADE)
        )
        df.loc[mascara_rf_sensibilidade, "resultado_oficial"] = False
        df.loc[mascara_rf_sensibilidade, "papel_experimento"] = "analise_sensibilidade"

        mascara_sensibilidade = df["conjunto_features"].eq(CONJUNTO_SENSIBILIDADE)
        df.loc[mascara_sensibilidade, "resultado_oficial"] = False

    registros = []
    for _, linha in df.iterrows():
        valores = linha.to_dict()
        metricas = calcular_metricas_confusao(
            int(valores["tn"]),
            int(valores["fp"]),
            int(valores["fn"]),
            int(valores["tp"]),
        )
        valores.update(metricas)
        registros.append(valores)

    return pd.DataFrame(registros)


def criar_linha_baseline_sempre_contaminada() -> pd.DataFrame:
    linha = {
        "modelo": "baseline_sempre_contaminada",
        "familia_modelo": "controle",
        "tipo_entrada": "controle",
        "cenario": "teste_baseline_sempre_contaminada",
        "threshold": "nao_aplicavel",
        "usa_pixels": False,
        "usa_recorte": False,
        "usa_atributos_visuais": False,
        "usa_metadados": False,
        "conjunto_features": CONJUNTO_NAO_APLICAVEL,
        "resultado_oficial": False,
        "papel_experimento": "controle",
        "arquivo_origem": "controle_gerado_no_script_27",
        "tn": 0,
        "fp": SUPORTE_NAO_CONTAMINADA_ESPERADO,
        "fn": 0,
        "tp": SUPORTE_CONTAMINADA_ESPERADO,
    }
    linha.update(calcular_metricas_confusao(linha["tn"], linha["fp"], linha["fn"], linha["tp"]))
    return pd.DataFrame([linha])


def carregar_comparacao() -> pd.DataFrame:
    tabelas = []
    for item in ARQUIVOS_METRICAS:
        df = ler_metricas_obrigatorio(item)
        tabelas.append(enriquecer_metricas(df, item))

    tabelas.append(criar_linha_baseline_sempre_contaminada())
    comparacao = pd.concat(tabelas, ignore_index=True, sort=False)

    for coluna in COLUNAS_SAIDA:
        if coluna not in comparacao.columns:
            comparacao[coluna] = np.nan

    extras = [coluna for coluna in comparacao.columns if coluna not in COLUNAS_SAIDA]
    return comparacao[COLUNAS_SAIDA + extras]


def validar_comparacao(df: pd.DataFrame):
    erros = []

    for coluna in COLUNAS_CONFUSAO:
        if df[coluna].isna().any():
            erros.append(f"Coluna {coluna} tem valores ausentes.")

    soma_confusao = df[COLUNAS_CONFUSAO].sum(axis=1)
    if not soma_confusao.eq(TOTAL_TESTE_ESPERADO).all():
        linhas = df.loc[~soma_confusao.eq(TOTAL_TESTE_ESPERADO), ["modelo", "cenario"]]
        erros.append(
            "Ha linhas em que tn+fp+fn+tp difere de "
            f"{TOTAL_TESTE_ESPERADO}: {linhas.to_dict(orient='records')}"
        )

    validacoes_suporte = {
        "total_teste": TOTAL_TESTE_ESPERADO,
        "suporte_contaminada": SUPORTE_CONTAMINADA_ESPERADO,
        "suporte_nao_contaminada": SUPORTE_NAO_CONTAMINADA_ESPERADO,
    }
    for coluna, esperado in validacoes_suporte.items():
        if not pd.to_numeric(df[coluna], errors="coerce").eq(esperado).all():
            linhas = df.loc[
                ~pd.to_numeric(df[coluna], errors="coerce").eq(esperado),
                ["modelo", "cenario", coluna],
            ]
            erros.append(
                f"Coluna {coluna} deveria ser {esperado} em todas as linhas: "
                f"{linhas.to_dict(orient='records')}"
            )

    duplicados = df[df.duplicated(["modelo", "conjunto_features", "cenario"], keep=False)]
    if not duplicados.empty:
        erros.append(
            "Ha duplicatas de modelo+conjunto_features+cenario: "
            f"{duplicados[['modelo', 'conjunto_features', 'cenario']].to_dict(orient='records')}"
        )

    diferenca_recall = (
        pd.to_numeric(df["recall_contaminada"], errors="coerce")
        - pd.to_numeric(df["sensibilidade_contaminada"], errors="coerce")
    ).abs()
    if not diferenca_recall.le(1e-9).all():
        erros.append("Recall e sensibilidade da classe contaminada nao coincidem.")

    for coluna in COLUNAS_METRICAS_TAXA:
        valores = pd.to_numeric(df[coluna], errors="coerce")
        invalidos = valores.isna() | (valores < -1e-9) | (valores > 1 + 1e-9)
        if invalidos.any():
            erros.append(
                f"Coluna {coluna} deveria estar entre 0 e 1: "
                f"{df.loc[invalidos, ['modelo', 'cenario', coluna]].to_dict(orient='records')}"
            )

    for coluna in ["youden_j", "mcc"]:
        valores = pd.to_numeric(df[coluna], errors="coerce")
        invalidos = valores.isna() | (valores < -1 - 1e-9) | (valores > 1 + 1e-9)
        if invalidos.any():
            erros.append(
                f"Coluna {coluna} deveria estar entre -1 e 1: "
                f"{df.loc[invalidos, ['modelo', 'cenario', coluna]].to_dict(orient='records')}"
            )

    if erros:
        raise ValueError("Validacao da comparacao final falhou:\n- " + "\n- ".join(erros))


def filtrar_candidatos_visuais(df: pd.DataFrame, cenario: str) -> pd.DataFrame:
    return df[
        (df["resultado_oficial"].astype(bool))
        & (df["cenario"].astype(str) == cenario)
        & (~df["usa_metadados"].astype(bool))
        & (df["tipo_entrada"].astype(str) != "controle")
        & (df["modelo"].astype(str) != "baseline_sempre_contaminada")
    ].copy()


def gerar_rankings(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_equilibrado = filtrar_candidatos_visuais(df, "teste_threshold_0_50")
    ranking_equilibrado = ranking_equilibrado.sort_values(
        ["balanced_accuracy", "mcc", "f1_contaminada", "recall_contaminada"],
        ascending=[False, False, False, False],
    )

    ranking_recall = filtrar_candidatos_visuais(
        df,
        "teste_threshold_prioridade_recall_validacao",
    )
    ranking_recall = ranking_recall.sort_values(
        [
            "recall_contaminada",
            "especificidade_nao_contaminada",
            "f1_contaminada",
            "balanced_accuracy",
        ],
        ascending=[False, False, False, False],
    )

    return ranking_equilibrado, ranking_recall


def formatar_linha(linha: pd.Series) -> str:
    return (
        f"{linha['modelo']} | cenario={linha['cenario']} | "
        f"features={linha['conjunto_features']} | "
        f"recall={linha['recall_contaminada']:.3f} | "
        f"especificidade={linha['especificidade_nao_contaminada']:.3f} | "
        f"F1={linha['f1_contaminada']:.3f} | "
        f"balanced_accuracy={linha['balanced_accuracy']:.3f} | "
        f"MCC={linha['mcc']:.3f}"
    )


def melhor_metadados(df: pd.DataFrame) -> pd.Series | None:
    metadados = df[df["usa_metadados"].astype(bool)].copy()
    if metadados.empty:
        return None
    return metadados.sort_values(
        ["f1_contaminada", "recall_contaminada", "especificidade_nao_contaminada"],
        ascending=[False, False, False],
    ).iloc[0]


def gerar_resumo(
    df: pd.DataFrame,
    ranking_equilibrado: pd.DataFrame,
    ranking_recall: pd.DataFrame,
) -> str:
    linhas = [
        "Resumo da comparacao final de classificacao",
        "=" * 48,
        "",
    ]

    if ranking_equilibrado.empty:
        linhas.append("Melhor modelo visual equilibrado: nao disponivel.")
    else:
        melhor_equilibrado = ranking_equilibrado.iloc[0]
        linhas.append("Melhor modelo visual equilibrado:")
        linhas.append(formatar_linha(melhor_equilibrado))

    linhas.append("")

    if ranking_recall.empty:
        linhas.append("Melhor modelo visual com prioridade de recall: nao disponivel.")
    else:
        melhor_recall = ranking_recall.iloc[0]
        linhas.append("Melhor modelo visual com prioridade de recall:")
        linhas.append(formatar_linha(melhor_recall))

    linhas.append("")
    sempre = df[df["modelo"].astype(str) == "baseline_sempre_contaminada"].iloc[0]
    linhas.append("Comparacao com baseline sempre contaminada:")
    linhas.append(formatar_linha(sempre))
    if not ranking_recall.empty:
        melhor_recall = ranking_recall.iloc[0]
        delta_especificidade = (
            melhor_recall["especificidade_nao_contaminada"]
            - sempre["especificidade_nao_contaminada"]
        )
        delta_balanced = melhor_recall["balanced_accuracy"] - sempre["balanced_accuracy"]
        linhas.append(
            "O melhor modelo visual de prioridade de recall ganha "
            f"{delta_especificidade:.3f} em especificidade e "
            f"{delta_balanced:.3f} em balanced accuracy contra esse controle."
        )

    linhas.append("")
    linha_metadados = melhor_metadados(df)
    linhas.append("Comparacao com baseline de metadados:")
    if linha_metadados is None:
        linhas.append("Baseline de metadados nao encontrado.")
    else:
        linhas.append(formatar_linha(linha_metadados))
        if not ranking_equilibrado.empty:
            melhor_equilibrado = ranking_equilibrado.iloc[0]
            linhas.append(
                "Diferenca do melhor visual equilibrado contra metadados: "
                f"F1={melhor_equilibrado['f1_contaminada'] - linha_metadados['f1_contaminada']:.3f}, "
                f"balanced_accuracy={melhor_equilibrado['balanced_accuracy'] - linha_metadados['balanced_accuracy']:.3f}, "
                f"MCC={melhor_equilibrado['mcc'] - linha_metadados['mcc']:.3f}."
            )

    candidatos_vies_operacional = df[
        (df["resultado_oficial"].astype(bool))
        & (~df["usa_metadados"].astype(bool))
        & (df["recall_contaminada"] >= 0.95)
        & (df["especificidade_nao_contaminada"] <= 0.10)
    ].copy()

    linhas.append("")
    linhas.append("Modelos com recall alto e especificidade quase nula:")
    if candidatos_vies_operacional.empty:
        linhas.append("Nenhum resultado oficial visual atingiu recall >= 0.95 com especificidade <= 0.10.")
    else:
        for _, linha in candidatos_vies_operacional.sort_values(
            ["recall_contaminada", "f1_contaminada"],
            ascending=[False, False],
        ).iterrows():
            linhas.append(formatar_linha(linha))

    linhas.extend([
        "",
        "Notas de interpretacao:",
        "- O baseline de metadados e diagnostico de vies de lote/tratamento, nao candidato visual para aplicativo.",
        "- O Random Forest com sensibilidade_todos_atributos e analise de sensibilidade, nao resultado oficial.",
        "- Resultados com recall alto e especificidade muito baixa podem estar proximos do controle sempre contaminada.",
        "- Esta comparacao nao prova generalizacao definitiva; o script 28 deve validar por tratamento/lote.",
    ])

    return "\n".join(linhas)


def main():
    print("=" * 60)
    print("COMPARANDO CLASSIFICACAO FINAL")
    print("=" * 60)

    PASTA_CLASSIFICACAO_FINAL.mkdir(parents=True, exist_ok=True)

    comparacao = carregar_comparacao()
    validar_comparacao(comparacao)
    ranking_equilibrado, ranking_recall = gerar_rankings(comparacao)
    resumo = gerar_resumo(comparacao, ranking_equilibrado, ranking_recall)

    comparacao.to_csv(CAMINHO_COMPARACAO_FINAL, index=False, encoding="utf-8-sig")
    ranking_equilibrado.to_csv(CAMINHO_RANKING_EQUILIBRADO, index=False, encoding="utf-8-sig")
    ranking_recall.to_csv(CAMINHO_RANKING_RECALL, index=False, encoding="utf-8-sig")
    CAMINHO_RESUMO.write_text(resumo, encoding="utf-8")

    print()
    print("Arquivos gerados:")
    for caminho in [
        CAMINHO_COMPARACAO_FINAL,
        CAMINHO_RANKING_EQUILIBRADO,
        CAMINHO_RANKING_RECALL,
        CAMINHO_RESUMO,
    ]:
        print(f"- {caminho}")


if __name__ == "__main__":
    main()
