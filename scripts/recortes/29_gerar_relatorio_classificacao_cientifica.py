from pathlib import Path
from datetime import datetime
import json
import shutil
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# SCRIPT 29 - GERAR RELATORIO CIENTIFICO DA CLASSIFICACAO
# ------------------------------------------------------------
# Objetivo:
# - Consolidar experimentos de classificacao ja concluidos
# - Gerar relatorio cientifico reproduzivel
# - Nao treinar modelos, nao recalibrar thresholds e nao alterar CSVs existentes
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_DOCS = PASTA_PROJETO / "docs"
PASTA_CLASSIFICACAO_FINAL = (
    PASTA_PROJETO / "saidas" / "tabelas" / "07_classificacao_final"
)
PASTA_VALIDACAO = PASTA_CLASSIFICACAO_FINAL / "validacao_tratamento"
PASTA_RELATORIO = PASTA_CLASSIFICACAO_FINAL / "relatorio"
PASTA_FIGURAS = PASTA_RELATORIO / "figuras"
PASTA_FIGURAS_DOCS = PASTA_DOCS / "figuras" / "classificacao"

CAMINHO_COMPARACAO_FINAL = PASTA_CLASSIFICACAO_FINAL / "comparacao_final_classificacao.csv"
CAMINHO_RESUMO_COMPARACAO = PASTA_CLASSIFICACAO_FINAL / "resumo_comparacao_classificacao.txt"
CAMINHO_RESUMO_VALIDACAO = PASTA_VALIDACAO / "resumo_generalizacao_por_tratamento.csv"
CAMINHO_METRICAS_VALIDACAO = PASTA_VALIDACAO / "metricas_validacao_por_tratamento.csv"
CAMINHO_COMPARACAO_PROTOCOLOS = PASTA_VALIDACAO / "comparacao_split_original_vs_tratamento.csv"
CAMINHO_DIAGNOSTICO_FOLDS = PASTA_VALIDACAO / "diagnostico_folds_validacao_por_tratamento.csv"
CAMINHO_CONFIG_VALIDACAO = PASTA_VALIDACAO / "config_validacao_por_tratamento.json"

CAMINHO_RELATORIO_MD = PASTA_DOCS / "relatorio_classificacao_cientifica.md"
CAMINHO_TABELA_SPLIT = PASTA_RELATORIO / "tabela_resultados_split_original.csv"
CAMINHO_TABELA_VALIDACAO = PASTA_RELATORIO / "tabela_resultados_validacao_externa.csv"
CAMINHO_TABELA_TRATAMENTO = PASTA_RELATORIO / "tabela_desempenho_por_tratamento.csv"
CAMINHO_MANIFESTO = PASTA_RELATORIO / "manifesto_experimento_final.json"

FIGURA_SPLIT_METRICAS = PASTA_FIGURAS / "metricas_split_original.png"
FIGURA_VALIDACAO_METRICAS = PASTA_FIGURAS / "metricas_validacao_externa_micro.png"
FIGURA_VARIACAO_TRATAMENTOS = PASTA_FIGURAS / "variacao_entre_tratamentos.png"
FIGURA_COMPARACAO_PROTOCOLOS = PASTA_FIGURAS / "comparacao_split_original_vs_validacao_externa.png"

METRICAS_PRIORITARIAS = [
    "balanced_accuracy",
    "mcc",
    "recall_contaminada",
    "especificidade_nao_contaminada",
]

NOMES_METRICAS = {
    "balanced_accuracy": "Balanced accuracy",
    "mcc": "MCC",
    "recall_contaminada": "Recall contaminada",
    "especificidade_nao_contaminada": "Especificidade",
    "delta_balanced_accuracy": "Delta balanced accuracy",
    "delta_mcc": "Delta MCC",
    "delta_recall_contaminada": "Delta recall",
    "delta_especificidade_nao_contaminada": "Delta especificidade",
}

CENARIO_EQUILIBRADO = "teste_threshold_0_50"
CENARIO_RECALL = "teste_threshold_prioridade_recall_validacao"
CENARIO_CONTROLE = "teste_baseline_sempre_contaminada"

MODELOS_METADADOS = {"metadados_taxas_suavizadas"}
MODELOS_CONTROLE = {"baseline_sempre_contaminada"}

NOMES_MODELOS = {
    "baseline_sempre_contaminada": "Controle: sempre contaminada",
    "metadados_taxas_suavizadas": "Metadados",
    "mobilenetv2_recortes": "MobileNetV2",
    "random_forest": "Random Forest",
    "svm_rbf": "SVM RBF",
    "knn": "k-NN",
    "lda": "LDA",
    "recortes_resnet18": "ResNet18 com recortes",
    "baseline_resnet18_imagem_inteira": "ResNet18 com imagem inteira",
    "yolo_caixas_automaticas": "YOLO com caixas automáticas",
}

COLUNAS_INTEIRAS = {
    "fold",
    "folds",
    "n_teste",
    "n_treino",
    "n_validacao",
    "total",
    "total_teste",
    "suporte_contaminada",
    "suporte_nao_contaminada",
    "tn",
    "fp",
    "fn",
    "tp",
    "teste_contaminada",
    "teste_nao_contaminada",
    "treino_contaminada",
    "treino_nao_contaminada",
    "validacao_contaminada",
    "validacao_nao_contaminada",
}

def caminho_relativo(caminho: Path) -> str:
    return str(caminho.relative_to(PASTA_PROJETO))


def caminho_relativo_docs(caminho: Path) -> str:
    return str(caminho.relative_to(PASTA_DOCS)).replace("\\", "/")


def nome_modelo_legivel(modelo: str) -> str:
    return NOMES_MODELOS.get(str(modelo), str(modelo))


def ler_csv_obrigatorio(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(
            "Arquivo obrigatorio ausente para o relatorio cientifico: "
            f"{caminho}"
        )
    df = pd.read_csv(caminho)
    if df.empty:
        raise ValueError(f"Arquivo obrigatorio vazio: {caminho}")
    return df


def ler_texto_obrigatorio(caminho: Path) -> str:
    if not caminho.exists():
        raise FileNotFoundError(
            "Texto obrigatorio ausente para o relatorio cientifico: "
            f"{caminho}"
        )
    texto = caminho.read_text(encoding="utf-8").strip()
    if not texto:
        raise ValueError(f"Texto obrigatorio vazio: {caminho}")
    return texto


def ler_json_obrigatorio(caminho: Path) -> dict:
    if not caminho.exists():
        raise FileNotFoundError(
            "Config obrigatoria ausente para o relatorio cientifico: "
            f"{caminho}"
        )
    return json.loads(caminho.read_text(encoding="utf-8"))


def carregar_artefatos() -> dict:
    return {
        "comparacao_final": ler_csv_obrigatorio(CAMINHO_COMPARACAO_FINAL),
        "resumo_comparacao": ler_texto_obrigatorio(CAMINHO_RESUMO_COMPARACAO),
        "resumo_validacao": ler_csv_obrigatorio(CAMINHO_RESUMO_VALIDACAO),
        "metricas_validacao": ler_csv_obrigatorio(CAMINHO_METRICAS_VALIDACAO),
        "comparacao_protocolos": ler_csv_obrigatorio(CAMINHO_COMPARACAO_PROTOCOLOS),
        "diagnostico_folds": ler_csv_obrigatorio(CAMINHO_DIAGNOSTICO_FOLDS),
        "config_validacao": ler_json_obrigatorio(CAMINHO_CONFIG_VALIDACAO),
    }


def salvar_csv(df: pd.DataFrame, caminho: Path):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False, encoding="utf-8-sig")


def normalizar_booleano(serie: pd.Series) -> pd.Series:
    if serie.dtype == bool:
        return serie
    return serie.astype(str).str.lower().isin(["true", "1", "sim"])


def coluna_existente(df: pd.DataFrame, coluna: str, padrao=None) -> pd.Series:
    if coluna in df.columns:
        return df[coluna]
    return pd.Series([padrao] * len(df), index=df.index)


def selecionar_colunas(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    saida = df.copy()
    for coluna in colunas:
        if coluna not in saida.columns:
            saida[coluna] = np.nan
    return saida[colunas]


def preparar_tabelas(artefatos: dict) -> dict:
    comparacao = artefatos["comparacao_final"].copy()
    resumo_validacao = artefatos["resumo_validacao"].copy()
    metricas_validacao = artefatos["metricas_validacao"].copy()

    comparacao["protocolo"] = "split_original_treino_validacao_teste"
    resumo_validacao["protocolo"] = "leave_one_experimento_tratamento_out"
    metricas_validacao["protocolo"] = "leave_one_experimento_tratamento_out"

    tabela_split = selecionar_colunas(
        comparacao,
        [
            "protocolo",
            "modelo",
            "familia_modelo",
            "tipo_entrada",
            "cenario",
            "threshold",
            "conjunto_features",
            "resultado_oficial",
            "papel_experimento",
            "balanced_accuracy",
            "mcc",
            "recall_contaminada",
            "especificidade_nao_contaminada",
            "f1_contaminada",
            "taxa_predita_contaminada",
            "tn",
            "fp",
            "fn",
            "tp",
            "total_teste",
            "suporte_contaminada",
            "suporte_nao_contaminada",
        ],
    )

    tabela_validacao = selecionar_colunas(
        resumo_validacao,
        [
            "protocolo",
            "agregacao",
            "modelo",
            "familia_modelo",
            "tipo_entrada",
            "cenario",
            "conjunto_features",
            "resultado_oficial",
            "papel_experimento",
            "folds",
            "balanced_accuracy",
            "mcc",
            "recall_contaminada",
            "especificidade_nao_contaminada",
            "f1_contaminada",
            "balanced_accuracy_media",
            "balanced_accuracy_dp",
            "mcc_media",
            "mcc_dp",
            "recall_contaminada_media",
            "recall_contaminada_dp",
            "especificidade_nao_contaminada_media",
            "especificidade_nao_contaminada_dp",
            "f1_contaminada_media",
            "f1_contaminada_dp",
            "tn",
            "fp",
            "fn",
            "tp",
            "total",
        ],
    )

    tabela_tratamento = selecionar_colunas(
        metricas_validacao,
        [
            "protocolo",
            "fold",
            "grupo_externo",
            "grupo_validacao",
            "modelo",
            "familia_modelo",
            "tipo_entrada",
            "cenario",
            "threshold",
            "conjunto_features",
            "resultado_oficial",
            "papel_experimento",
            "balanced_accuracy",
            "mcc",
            "recall_contaminada",
            "especificidade_nao_contaminada",
            "f1_contaminada",
            "taxa_predita_contaminada",
            "tn",
            "fp",
            "fn",
            "tp",
            "total",
            "suporte_contaminada",
            "suporte_nao_contaminada",
            "tempo_treino_segundos",
            "melhor_epoca",
        ],
    )

    salvar_csv(tabela_split, CAMINHO_TABELA_SPLIT)
    salvar_csv(tabela_validacao, CAMINHO_TABELA_VALIDACAO)
    salvar_csv(tabela_tratamento, CAMINHO_TABELA_TRATAMENTO)

    return {
        "tabela_split": tabela_split,
        "tabela_validacao": tabela_validacao,
        "tabela_tratamento": tabela_tratamento,
    }


def rotulo_linha(linha: pd.Series) -> str:
    return nome_modelo_legivel(str(linha.get("modelo", "modelo")))


def filtrar_para_figura_split(df: pd.DataFrame) -> pd.DataFrame:
    trabalho = df.copy()
    if "resultado_oficial" in trabalho.columns:
        oficial = normalizar_booleano(trabalho["resultado_oficial"])
    else:
        oficial = pd.Series([False] * len(trabalho), index=trabalho.index)
    modelos_controle = trabalho["modelo"].astype(str).isin(
        ["baseline_sempre_contaminada", "metadados_taxas_suavizadas"]
    )
    mascara = (
        trabalho["cenario"].astype(str).isin([CENARIO_EQUILIBRADO, CENARIO_CONTROLE])
        & (oficial | modelos_controle)
    )
    return trabalho[mascara].copy()


def filtrar_validacao_micro(df: pd.DataFrame) -> pd.DataFrame:
    trabalho = df.copy()
    return trabalho[
        (trabalho["agregacao"].astype(str) == "micro")
        & (trabalho["cenario"].astype(str).isin([CENARIO_EQUILIBRADO, CENARIO_CONTROLE]))
    ].copy()


def filtrar_validacao_micro_todos_cenarios(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["agregacao"].astype(str) == "micro"].copy()


def filtrar_modelos_visuais(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    modelos = df["modelo"].astype(str)
    visual = ~(modelos.isin(MODELOS_METADADOS | MODELOS_CONTROLE))
    if "papel_experimento" in df.columns:
        visual &= df["papel_experimento"].astype(str) != "diagnostico_vies"
    return df[visual].copy()


def plotar_metricas_barras(df: pd.DataFrame, caminho: Path, titulo: str):
    if df.empty:
        return

    dados = df.copy()
    dados["rotulo"] = dados.apply(rotulo_linha, axis=1)
    dados = dados.sort_values(["cenario", "modelo", "conjunto_features"])

    x = np.arange(len(dados))
    largura = 0.18
    fig, ax = plt.subplots(figsize=(max(10, len(dados) * 0.8), 6))

    for indice, metrica in enumerate(METRICAS_PRIORITARIAS):
        valores = pd.to_numeric(dados[metrica], errors="coerce").fillna(0.0)
        deslocamento = (indice - 1.5) * largura
        ax.bar(
            x + deslocamento,
            valores,
            width=largura,
            label=NOMES_METRICAS.get(metrica, metrica),
        )

    ax.set_title(titulo)
    ax.set_ylabel("Métrica")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(dados["rotulo"], rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=160)
    plt.close(fig)


def plotar_variacao_tratamentos(df: pd.DataFrame, caminho: Path):
    dados = df[
        df["cenario"].astype(str).isin([CENARIO_EQUILIBRADO, CENARIO_CONTROLE])
    ].copy()
    if dados.empty:
        return

    if "resultado_oficial" in dados.columns:
        oficial = normalizar_booleano(dados["resultado_oficial"])
    else:
        oficial = pd.Series([False] * len(dados), index=dados.index)
    controles = dados["modelo"].astype(str).isin(
        ["baseline_sempre_contaminada", "metadados_taxas_suavizadas"]
    )
    dados = dados[oficial | controles].copy()
    if dados.empty:
        return

    grupos = sorted(dados["modelo"].astype(str).unique())
    rotulos = [nome_modelo_legivel(modelo) for modelo in grupos]
    series = [
        pd.to_numeric(
            dados[dados["modelo"].astype(str) == modelo]["balanced_accuracy"],
            errors="coerce",
        ).dropna()
        for modelo in grupos
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(grupos) * 0.9), 6))
    ax.boxplot(series, tick_labels=rotulos, showmeans=True)
    ax.set_title("Variação de balanced accuracy entre tratamentos")
    ax.set_ylabel("Balanced accuracy por grupo externo")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=160)
    plt.close(fig)


def plotar_comparacao_protocolos(df: pd.DataFrame, caminho: Path):
    if df.empty:
        return

    dados = df[df["cenario"].astype(str) == CENARIO_EQUILIBRADO].copy()
    if dados.empty:
        return

    if "delta_balanced_accuracy" not in dados.columns:
        return

    dados["rotulo"] = dados.apply(rotulo_linha, axis=1)
    dados = dados.sort_values(["cenario", "modelo", "conjunto_features"])
    metricas_delta = [
        "delta_balanced_accuracy",
        "delta_mcc",
        "delta_recall_contaminada",
        "delta_especificidade_nao_contaminada",
    ]
    metricas_delta = [metrica for metrica in metricas_delta if metrica in dados.columns]

    x = np.arange(len(dados))
    largura = 0.18
    fig, ax = plt.subplots(figsize=(max(10, len(dados) * 0.8), 6))
    for indice, metrica in enumerate(metricas_delta):
        valores = pd.to_numeric(dados[metrica], errors="coerce").fillna(0.0)
        deslocamento = (indice - (len(metricas_delta) - 1) / 2) * largura
        ax.bar(
            x + deslocamento,
            valores,
            width=largura,
            label=NOMES_METRICAS.get(metrica, metrica),
        )

    ax.set_title("Diferença: validação externa - split original")
    ax.set_ylabel("Delta da métrica")
    ax.set_xticks(x)
    ax.set_xticklabels(dados["rotulo"], rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=160)
    plt.close(fig)


def copiar_figuras_para_docs(figuras: list[Path]) -> list[Path]:
    PASTA_FIGURAS_DOCS.mkdir(parents=True, exist_ok=True)
    figuras_docs = []
    for figura in figuras:
        destino = PASTA_FIGURAS_DOCS / figura.name
        shutil.copy2(figura, destino)
        figuras_docs.append(destino)
    return figuras_docs


def gerar_figuras(tabelas: dict, artefatos: dict) -> dict[str, list[Path]]:
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    figuras = []

    plotar_metricas_barras(
        filtrar_para_figura_split(tabelas["tabela_split"]),
        FIGURA_SPLIT_METRICAS,
        "Split original: balanced accuracy, MCC, recall e especificidade",
    )
    figuras.append(FIGURA_SPLIT_METRICAS)

    plotar_metricas_barras(
        filtrar_validacao_micro(tabelas["tabela_validacao"]),
        FIGURA_VALIDACAO_METRICAS,
        "Validação externa micro: balanced accuracy, MCC, recall e especificidade",
    )
    figuras.append(FIGURA_VALIDACAO_METRICAS)

    plotar_variacao_tratamentos(tabelas["tabela_tratamento"], FIGURA_VARIACAO_TRATAMENTOS)
    figuras.append(FIGURA_VARIACAO_TRATAMENTOS)

    plotar_comparacao_protocolos(
        artefatos["comparacao_protocolos"],
        FIGURA_COMPARACAO_PROTOCOLOS,
    )
    figuras.append(FIGURA_COMPARACAO_PROTOCOLOS)

    figuras_saida = [figura for figura in figuras if figura.exists()]
    figuras_docs = copiar_figuras_para_docs(figuras_saida)

    return {
        "saida": figuras_saida,
        "docs": figuras_docs,
    }


def melhor_por_metrica(df: pd.DataFrame, metrica: str) -> pd.Series | None:
    if df.empty or metrica not in df.columns:
        return None
    dados = df.copy()
    dados[metrica] = pd.to_numeric(dados[metrica], errors="coerce")
    dados = dados.dropna(subset=[metrica])
    if dados.empty:
        return None
    return dados.sort_values(
        [metrica, "mcc", "recall_contaminada", "especificidade_nao_contaminada"],
        ascending=[False, False, False, False],
    ).iloc[0]


def exigir_resultado_unico(
    df: pd.DataFrame,
    descricao: str,
    criterios: dict[str, str],
) -> pd.Series:
    mascara = pd.Series([True] * len(df), index=df.index)

    for coluna, valor in criterios.items():
        if coluna not in df.columns:
            raise ValueError(
                f"Coluna obrigatória ausente para localizar {descricao}: {coluna}"
            )
        mascara &= df[coluna].astype(str) == str(valor)

    linhas = df[mascara]
    if len(linhas) != 1:
        criterios_txt = ", ".join(
            f"{coluna}={valor!r}" for coluna, valor in criterios.items()
        )
        raise ValueError(
            f"Resultado esperado com exatamente uma linha para {descricao}; "
            f"encontradas {len(linhas)} linhas. Critérios: {criterios_txt}"
        )

    return linhas.iloc[0]


def formatar_numero(valor, casas: int = 3) -> str:
    if pd.isna(valor):
        return "NA"
    try:
        return f"{float(valor):.{casas}f}"
    except (TypeError, ValueError):
        return str(valor)


def formatar_inteiro(valor) -> str:
    if pd.isna(valor):
        return "NA"
    try:
        return str(int(round(float(valor))))
    except (TypeError, ValueError):
        return str(valor)


def formatar_valor_tabela(valor, coluna: str) -> str:
    if coluna in COLUNAS_INTEIRAS:
        return formatar_inteiro(valor)
    return formatar_numero(valor)


def formatar_linha_resultado(linha: pd.Series | None) -> str:
    if linha is None:
        return "Não disponível."
    return (
        f"{nome_modelo_legivel(linha.get('modelo', 'modelo'))} | "
        f"cenário={linha.get('cenario', 'NA')} | "
        f"features={linha.get('conjunto_features', 'NA')} | "
        f"balanced_accuracy={formatar_numero(linha.get('balanced_accuracy'))} | "
        f"MCC={formatar_numero(linha.get('mcc'))} | "
        f"recall={formatar_numero(linha.get('recall_contaminada'))} | "
        f"especificidade={formatar_numero(linha.get('especificidade_nao_contaminada'))} | "
        f"F1={formatar_numero(linha.get('f1_contaminada'))}"
    )


def tabela_markdown(df: pd.DataFrame, colunas: list[str], max_linhas: int = 8) -> str:
    if df.empty:
        return "Sem dados disponíveis."
    dados = selecionar_colunas(df, colunas).head(max_linhas).copy()
    for coluna in dados.columns:
        if pd.api.types.is_numeric_dtype(dados[coluna]):
            dados[coluna] = dados[coluna].map(
                lambda valor, nome_coluna=coluna: formatar_valor_tabela(
                    valor, nome_coluna
                )
            )
        else:
            dados[coluna] = dados[coluna].fillna("NA").astype(str)

    cabecalho = "| " + " | ".join(dados.columns) + " |"
    separador = "| " + " | ".join(["---"] * len(dados.columns)) + " |"
    linhas = [
        "| " + " | ".join(str(valor) for valor in linha) + " |"
        for linha in dados.to_numpy()
    ]
    return "\n".join([cabecalho, separador, *linhas])


def resumir_grupos(diagnostico: pd.DataFrame, config: dict) -> dict:
    total_amostras = int(config.get("total_amostras", 703))
    total_grupos = int(config.get("total_grupos", diagnostico["grupo_externo"].nunique()))
    grupos_pequenos = diagnostico[diagnostico["n_teste"] < 10].copy()
    menor_grupo = diagnostico.sort_values("n_teste").head(1)
    maior_grupo = diagnostico.sort_values("n_teste", ascending=False).head(1)
    return {
        "total_amostras": total_amostras,
        "total_grupos": total_grupos,
        "grupos_pequenos": grupos_pequenos,
        "menor_grupo": menor_grupo,
        "maior_grupo": maior_grupo,
    }


def criar_manifesto(artefatos: dict, tabelas: dict, figuras: dict[str, list[Path]]) -> dict:
    config = artefatos["config_validacao"]
    modelos_concluidos = sorted(
        artefatos["metricas_validacao"]["modelo"].dropna().astype(str).unique()
    )
    manifesto = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "objetivo": "relatório científico final da classificação",
        "nao_executa_treino": True,
        "nao_recalibra_thresholds": True,
        "arquivos_lidos": {
            "comparacao_final": caminho_relativo(CAMINHO_COMPARACAO_FINAL),
            "resumo_comparacao": caminho_relativo(CAMINHO_RESUMO_COMPARACAO),
            "resumo_validacao": caminho_relativo(CAMINHO_RESUMO_VALIDACAO),
            "metricas_validacao": caminho_relativo(CAMINHO_METRICAS_VALIDACAO),
            "comparacao_protocolos": caminho_relativo(CAMINHO_COMPARACAO_PROTOCOLOS),
            "diagnostico_folds": caminho_relativo(CAMINHO_DIAGNOSTICO_FOLDS),
            "config_validacao": caminho_relativo(CAMINHO_CONFIG_VALIDACAO),
        },
        "arquivos_gerados": {
            "relatorio_markdown": caminho_relativo(CAMINHO_RELATORIO_MD),
            "tabela_split_original": caminho_relativo(CAMINHO_TABELA_SPLIT),
            "tabela_validacao_externa": caminho_relativo(CAMINHO_TABELA_VALIDACAO),
            "tabela_por_tratamento": caminho_relativo(CAMINHO_TABELA_TRATAMENTO),
            "manifesto": caminho_relativo(CAMINHO_MANIFESTO),
            "figuras_saida": [
                caminho_relativo(figura) for figura in figuras.get("saida", [])
            ],
            "figuras_docs": [
                caminho_relativo(figura) for figura in figuras.get("docs", [])
            ],
        },
        "protocolo_validacao_externa": config.get("protocolo"),
        "grupo_principal": config.get("grupo_principal"),
        "total_amostras": config.get("total_amostras"),
        "total_grupos": config.get("total_grupos"),
        "modelos_concluidos_validacao": modelos_concluidos,
        "modelos_solicitados_config_ultimo_registro": config.get("modelos_solicitados"),
        "parametros_principais": {
            "random_forest": config.get("random_forest"),
            "svm_rbf": config.get("svm_rbf"),
            "knn": config.get("knn"),
            "lda": config.get("lda"),
            "metadados": config.get("metadados"),
            "mobilenetv2": config.get("mobilenetv2"),
            "thresholds": config.get("thresholds"),
        },
    }
    CAMINHO_MANIFESTO.write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifesto


def gerar_relatorio_md(
    artefatos: dict,
    tabelas: dict,
    figuras: dict[str, list[Path]],
    manifesto: dict,
) -> str:
    split = tabelas["tabela_split"]
    validacao = tabelas["tabela_validacao"]
    diagnostico = artefatos["diagnostico_folds"]
    config = artefatos["config_validacao"]
    grupos = resumir_grupos(diagnostico, config)

    split_equilibrado = filtrar_para_figura_split(split)
    split_visual_threshold_0_50 = filtrar_modelos_visuais(
        split[split["cenario"].astype(str) == CENARIO_EQUILIBRADO]
    )
    validacao_micro_todos = filtrar_validacao_micro_todos_cenarios(validacao)
    validacao_micro_visual = filtrar_modelos_visuais(validacao_micro_todos)

    melhor_split_visual = melhor_por_metrica(
        split_visual_threshold_0_50, "balanced_accuracy"
    )
    melhor_ext_visual = melhor_por_metrica(validacao_micro_visual, "balanced_accuracy")
    baseline_split = split[split["modelo"].astype(str) == "baseline_sempre_contaminada"].copy()
    baseline_ext = validacao[validacao["modelo"].astype(str) == "baseline_sempre_contaminada"].copy()
    metadados_split = split[split["papel_experimento"].astype(str) == "diagnostico_vies"].copy()
    metadados_ext = validacao[validacao["papel_experimento"].astype(str) == "diagnostico_vies"].copy()

    grupos_pequenos = grupos["grupos_pequenos"]
    texto_grupos_pequenos = (
        "Não houve grupos externos com menos de 10 amostras no diagnóstico carregado."
        if grupos_pequenos.empty
        else tabela_markdown(
            grupos_pequenos,
            [
                "fold",
                "grupo_externo",
                "n_teste",
                "teste_contaminada",
                "teste_nao_contaminada",
            ],
            max_linhas=12,
        )
    )

    figuras_md = "\n".join(
        f"![{figura.stem}]({caminho_relativo_docs(figura)})"
        for figura in figuras.get("docs", [])
    )

    parametros = manifesto["parametros_principais"]
    parametros_txt = json.dumps(parametros, indent=2, ensure_ascii=False)
    modelos_concluidos = ", ".join(
        nome_modelo_legivel(modelo)
        for modelo in manifesto.get("modelos_concluidos_validacao", [])
    )
    mobile_split = exigir_resultado_unico(
        split,
        "MobileNetV2 no split original",
        {
            "modelo": "mobilenetv2_recortes",
            "cenario": CENARIO_EQUILIBRADO,
            "conjunto_features": "nao_aplicavel",
        },
    )
    mobile_externo = exigir_resultado_unico(
        validacao,
        "MobileNetV2 na validação externa micro",
        {
            "agregacao": "micro",
            "modelo": "mobilenetv2_recortes",
            "cenario": CENARIO_EQUILIBRADO,
            "conjunto_features": "nao_aplicavel",
        },
    )
    rf_externo = exigir_resultado_unico(
        validacao,
        "Random Forest na validação externa micro com threshold validado",
        {
            "agregacao": "micro",
            "modelo": "random_forest",
            "cenario": "teste_threshold_melhor_f1_validacao",
            "conjunto_features": "principal_normalizado",
        },
    )

    relatorio = f"""
# Relatório científico final da classificação

Gerado em: {manifesto['gerado_em']}

## 1. Objetivo da classificação

O objetivo da classificação é estimar, a partir de imagens iniciais e
experimentos associados, o risco de contaminação posterior em sementes de
macaúba. A classe positiva é `contaminada`. A interpretação científica não deve
ser de detecção visual direta de infecção, mas de predição de risco associada ao
resultado observado posteriormente.

## 2. Amostras e grupos experimentais

A validação externa foi configurada para {grupos['total_amostras']} amostras e
{grupos['total_grupos']} grupos `experimento_tratamento`. Esse grupo combina
`experimento_rotulo` e `tratamento_planilha` normalizados, reduzindo o risco de
que amostras do mesmo contexto experimental apareçam simultaneamente em treino
e teste externo.

Menor grupo externo:

{tabela_markdown(grupos['menor_grupo'], ['fold', 'grupo_externo', 'n_teste', 'teste_contaminada', 'teste_nao_contaminada'], 1)}

Maior grupo externo:

{tabela_markdown(grupos['maior_grupo'], ['fold', 'grupo_externo', 'n_teste', 'teste_contaminada', 'teste_nao_contaminada'], 1)}

## 3. Protocolo do split original

O split original usa a divisão treino/validação/teste consolidada em
`saidas/tabelas/04_dataset_split/divisao_treino_validacao_teste.csv`. Os
modelos e thresholds do split original foram gerados em etapas anteriores; este
script apenas lê `comparacao_final_classificacao.csv` e não recalcula
thresholds.

## 4. Protocolo leave-one-experimento-tratamento-out

Na validação externa, cada grupo `experimento_tratamento` é deixado de fora uma
vez como teste externo. A validação interna usa um grupo inteiro do conjunto de
desenvolvimento, escolhido deterministicamente. O split original permanece
apenas como coluna de auditoria.

## 5. Modelos avaliados

Foram consolidados modelos de imagem inteira, YOLO/caixas, ResNet18 com
recortes, Random Forest, SVM RBF, k-NN e LDA com atributos visuais
normalizados, MobileNetV2 com recortes, baseline de metadados e baseline
sempre-contaminada. O baseline de
metadados é tratado como diagnóstico de viés de lote/tratamento, não como
candidato visual para aplicativo. O baseline sempre-contaminada é um controle,
não um modelo operacional.

Modelos concluídos na validação externa: {modelos_concluidos}.

## 6. Parâmetros científicos principais

```json
{parametros_txt}
```

## 7. Resultados do split original

Melhor modelo visual no split original com `threshold=0,50`:

{formatar_linha_resultado(melhor_split_visual)}

O baseline de metadados pode aparecer acima de modelos visuais no split original,
mas essa linha é diagnóstica: ela indica que origem, tratamento, pasta e campos
derivados carregam informação sobre o lote/tratamento. Ela não é candidata ao
aplicativo.

Tabela resumida do split original no cenário `teste_threshold_0_50` e controle:

{tabela_markdown(split_equilibrado.sort_values(['balanced_accuracy', 'mcc'], ascending=[False, False]), ['modelo', 'cenario', 'conjunto_features', 'balanced_accuracy', 'mcc', 'recall_contaminada', 'especificidade_nao_contaminada', 'f1_contaminada'], 12)}

## 8. Resultados externos micro e macro

Na agregação micro, as matrizes de confusão dos folds são somadas antes do
cálculo das métricas. Ela pesa mais os grupos com mais amostras. Na agregação
macro, as métricas são calculadas por grupo externo e depois resumidas por média
e desvio-padrão. Essa leitura mostra variação entre tratamentos, mas fica
instável quando há grupos pequenos.

Melhor modelo visual na validação externa:

{formatar_linha_resultado(melhor_ext_visual)}

Resumo micro:

{tabela_markdown(validacao[validacao['agregacao'].astype(str) == 'micro'].sort_values(['balanced_accuracy', 'mcc'], ascending=[False, False]), ['modelo', 'cenario', 'conjunto_features', 'balanced_accuracy', 'mcc', 'recall_contaminada', 'especificidade_nao_contaminada', 'f1_contaminada', 'folds'], 12)}

Resumo macro:

{tabela_markdown(validacao[validacao['agregacao'].astype(str) == 'macro'].sort_values(['balanced_accuracy_media', 'mcc_media'], ascending=[False, False]), ['modelo', 'cenario', 'conjunto_features', 'balanced_accuracy_media', 'balanced_accuracy_dp', 'mcc_media', 'mcc_dp', 'recall_contaminada_media', 'especificidade_nao_contaminada_media', 'folds'], 12)}

## 9. Comparação com baseline sempre-contaminada

O baseline sempre-contaminada é um controle obrigatório: ele tende a maximizar
recall da classe contaminada ao custo de especificidade nula ou muito baixa.
Resultados com F1 alto devem ser interpretados contra esse controle.

Split original:

{tabela_markdown(baseline_split, ['cenario', 'balanced_accuracy', 'mcc', 'recall_contaminada', 'especificidade_nao_contaminada', 'f1_contaminada'], 5)}

Validação externa:

{tabela_markdown(baseline_ext, ['agregacao', 'cenario', 'balanced_accuracy', 'mcc', 'recall_contaminada', 'especificidade_nao_contaminada', 'f1_contaminada', 'balanced_accuracy_media', 'mcc_media'], 8)}

## 10. Diagnóstico de viés de lote/tratamento

O baseline de metadados usa origem, tratamento, pasta e campos derivados. Ele
serve para diagnosticar viés de lote/tratamento e não deve ser tratado como
modelo visual candidato ao aplicativo. Portanto, não há declaração de vencedor
baseada nos metadados, mesmo quando suas métricas superam as de modelos visuais.

Split original metadados:

{tabela_markdown(metadados_split, ['cenario', 'balanced_accuracy', 'mcc', 'recall_contaminada', 'especificidade_nao_contaminada', 'f1_contaminada'], 8)}

Validação externa metadados:

{tabela_markdown(metadados_ext, ['agregacao', 'cenario', 'balanced_accuracy', 'mcc', 'recall_contaminada', 'especificidade_nao_contaminada', 'f1_contaminada', 'balanced_accuracy_media', 'mcc_media'], 8)}

## 11. Limitações

As conclusões são limitadas pelo tamanho da base, pelo número de grupos
experimentais e por possíveis diferenças técnicas entre lotes, tratamentos,
origens e padrões de imagem. Grupos pequenos reduzem a estabilidade das métricas
por tratamento e tornam as médias macro mais instáveis, porque cada grupo
externo recebe o mesmo peso independentemente da quantidade de amostras.

Grupos pequenos no diagnóstico dos folds:

{texto_grupos_pequenos}

## 12. Conclusão sobre viabilidade da classificação

Não se deve declarar vencedor apenas por F1. A leitura prioritária combina
balanced accuracy, MCC, recall da classe contaminada e especificidade da classe
não contaminada.

No split original, o melhor modelo visual foi o MobileNetV2 com recortes no
`threshold=0,50`, com balanced accuracy {mobile_split['balanced_accuracy']:.3f}
e MCC {mobile_split['mcc']:.3f}. Na validação externa com o mesmo threshold, o
MobileNetV2 caiu para balanced accuracy
{mobile_externo['balanced_accuracy']:.3f} e MCC {mobile_externo['mcc']:.3f}. O
Random Forest externo com threshold validado obteve balanced accuracy
{rf_externo['balanced_accuracy']:.3f} e MCC {rf_externo['mcc']:.3f}.

Esses resultados indicam de forma afirmativa que nenhum modelo visual
generalizou de forma suficiente para classificação automática direta em
tratamentos desconhecidos. O desempenho externo fica próximo de um sinal fraco:
MCC negativo para MobileNetV2 no threshold fixo e MCC baixo para Random Forest
com threshold validado. A conclusão operacional é que a classificação direta
automática ainda não é viável como decisão final em lotes/tratamentos não
vistos.

## 13. Justificativa para avançar para triagem preventiva

A classificação direta exige boa sensibilidade sem destruir a especificidade. O
histórico dos experimentos mostra que recall alto pode ser obtido por regras
conservadoras próximas ao baseline sempre-contaminada. Portanto, a etapa
operacional mais defensável é triagem preventiva: separar alto risco, revisar
casos incertos e evitar liberação automática de baixo risco sem validação
adicional por lote/tratamento.

## Figuras

{figuras_md}

## Arquivos derivados deste relatório

- `{caminho_relativo(CAMINHO_TABELA_SPLIT)}`
- `{caminho_relativo(CAMINHO_TABELA_VALIDACAO)}`
- `{caminho_relativo(CAMINHO_TABELA_TRATAMENTO)}`
- `{caminho_relativo(CAMINHO_MANIFESTO)}`
"""
    return textwrap.dedent(relatorio).strip() + "\n"


def main():
    print("=" * 70)
    print("GERANDO RELATÓRIO CIENTÍFICO FINAL DA CLASSIFICAÇÃO")
    print("=" * 70)
    print("Este script não treina modelos e não recalibra thresholds.")

    PASTA_RELATORIO.mkdir(parents=True, exist_ok=True)
    PASTA_DOCS.mkdir(parents=True, exist_ok=True)

    artefatos = carregar_artefatos()
    tabelas = preparar_tabelas(artefatos)
    figuras = gerar_figuras(tabelas, artefatos)
    manifesto = criar_manifesto(artefatos, tabelas, figuras)
    relatorio = gerar_relatorio_md(artefatos, tabelas, figuras, manifesto)

    CAMINHO_RELATORIO_MD.write_text(relatorio, encoding="utf-8")

    print()
    print("Arquivos gerados:")
    for caminho in [
        CAMINHO_RELATORIO_MD,
        CAMINHO_TABELA_SPLIT,
        CAMINHO_TABELA_VALIDACAO,
        CAMINHO_TABELA_TRATAMENTO,
        CAMINHO_MANIFESTO,
        *figuras.get("saida", []),
        *figuras.get("docs", []),
    ]:
        print(f"- {caminho}")


if __name__ == "__main__":
    main()
