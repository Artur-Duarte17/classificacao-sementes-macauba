from pathlib import Path
from datetime import datetime
import json
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

CENARIO_EQUILIBRADO = "teste_threshold_0_50"
CENARIO_RECALL = "teste_threshold_prioridade_recall_validacao"
CENARIO_CONTROLE = "teste_baseline_sempre_contaminada"


def caminho_relativo(caminho: Path) -> str:
    return str(caminho.relative_to(PASTA_PROJETO))


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
    modelo = str(linha.get("modelo", "modelo"))
    features = str(linha.get("conjunto_features", ""))
    if features and features != "nao_aplicavel":
        return f"{modelo}\\n{features}"
    return modelo


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
        ax.bar(x + deslocamento, valores, width=largura, label=metrica)

    ax.set_title(titulo)
    ax.set_ylabel("Metrica")
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
    series = [
        pd.to_numeric(
            dados[dados["modelo"].astype(str) == modelo]["balanced_accuracy"],
            errors="coerce",
        ).dropna()
        for modelo in grupos
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(grupos) * 0.9), 6))
    ax.boxplot(series, labels=grupos, showmeans=True)
    ax.set_title("Variacao de balanced accuracy entre tratamentos")
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

    dados = df.copy()
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
        ax.bar(x + deslocamento, valores, width=largura, label=metrica)

    ax.set_title("Diferenca: validacao externa - split original")
    ax.set_ylabel("Delta da metrica")
    ax.set_xticks(x)
    ax.set_xticklabels(dados["rotulo"], rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=160)
    plt.close(fig)


def gerar_figuras(tabelas: dict, artefatos: dict) -> list[Path]:
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
        "Validacao externa micro: balanced accuracy, MCC, recall e especificidade",
    )
    figuras.append(FIGURA_VALIDACAO_METRICAS)

    plotar_variacao_tratamentos(tabelas["tabela_tratamento"], FIGURA_VARIACAO_TRATAMENTOS)
    figuras.append(FIGURA_VARIACAO_TRATAMENTOS)

    plotar_comparacao_protocolos(
        artefatos["comparacao_protocolos"],
        FIGURA_COMPARACAO_PROTOCOLOS,
    )
    figuras.append(FIGURA_COMPARACAO_PROTOCOLOS)

    return [figura for figura in figuras if figura.exists()]


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


def formatar_numero(valor, casas: int = 3) -> str:
    if pd.isna(valor):
        return "NA"
    try:
        return f"{float(valor):.{casas}f}"
    except (TypeError, ValueError):
        return str(valor)


def formatar_linha_resultado(linha: pd.Series | None) -> str:
    if linha is None:
        return "Nao disponivel."
    return (
        f"{linha.get('modelo', 'modelo')} | cenario={linha.get('cenario', 'NA')} | "
        f"features={linha.get('conjunto_features', 'NA')} | "
        f"balanced_accuracy={formatar_numero(linha.get('balanced_accuracy'))} | "
        f"MCC={formatar_numero(linha.get('mcc'))} | "
        f"recall={formatar_numero(linha.get('recall_contaminada'))} | "
        f"especificidade={formatar_numero(linha.get('especificidade_nao_contaminada'))} | "
        f"F1={formatar_numero(linha.get('f1_contaminada'))}"
    )


def tabela_markdown(df: pd.DataFrame, colunas: list[str], max_linhas: int = 8) -> str:
    if df.empty:
        return "Sem dados disponiveis."
    dados = selecionar_colunas(df, colunas).head(max_linhas).copy()
    for coluna in dados.columns:
        if pd.api.types.is_numeric_dtype(dados[coluna]):
            dados[coluna] = dados[coluna].map(lambda valor: formatar_numero(valor))
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


def criar_manifesto(artefatos: dict, tabelas: dict, figuras: list[Path]) -> dict:
    config = artefatos["config_validacao"]
    manifesto = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "objetivo": "relatorio cientifico final da classificacao",
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
            "figuras": [caminho_relativo(figura) for figura in figuras],
        },
        "protocolo_validacao_externa": config.get("protocolo"),
        "grupo_principal": config.get("grupo_principal"),
        "total_amostras": config.get("total_amostras"),
        "total_grupos": config.get("total_grupos"),
        "modelos_solicitados_validacao": config.get("modelos_solicitados"),
        "parametros_principais": {
            "random_forest": config.get("random_forest"),
            "svm_rbf": config.get("svm_rbf"),
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


def gerar_relatorio_md(artefatos: dict, tabelas: dict, figuras: list[Path], manifesto: dict) -> str:
    split = tabelas["tabela_split"]
    validacao = tabelas["tabela_validacao"]
    tratamentos = tabelas["tabela_tratamento"]
    diagnostico = artefatos["diagnostico_folds"]
    config = artefatos["config_validacao"]
    grupos = resumir_grupos(diagnostico, config)

    split_equilibrado = filtrar_para_figura_split(split)
    validacao_micro = filtrar_validacao_micro(validacao)

    melhor_split_bal = melhor_por_metrica(split_equilibrado, "balanced_accuracy")
    melhor_ext_bal = melhor_por_metrica(validacao_micro, "balanced_accuracy")
    baseline_split = split[split["modelo"].astype(str) == "baseline_sempre_contaminada"].copy()
    baseline_ext = validacao[validacao["modelo"].astype(str) == "baseline_sempre_contaminada"].copy()
    metadados_split = split[split["papel_experimento"].astype(str) == "diagnostico_vies"].copy()
    metadados_ext = validacao[validacao["papel_experimento"].astype(str) == "diagnostico_vies"].copy()

    grupos_pequenos = grupos["grupos_pequenos"]
    texto_grupos_pequenos = (
        "Nao houve grupos externos com menos de 10 amostras no diagnostico carregado."
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
        f"![{figura.stem}]({caminho_relativo(figura).replace(chr(92), '/')})"
        for figura in figuras
    )

    parametros = manifesto["parametros_principais"]
    parametros_txt = json.dumps(parametros, indent=2, ensure_ascii=False)

    relatorio = f"""
# Relatorio cientifico final da classificacao

Gerado em: {manifesto['gerado_em']}

## 1. Objetivo da classificacao

O objetivo da classificacao e estimar, a partir de imagens iniciais e
experimentos associados, o risco de contaminacao posterior em sementes de
macauba. A classe positiva e `contaminada`. A interpretacao cientifica nao deve
ser de deteccao visual direta de infeccao, mas de predicao de risco associada ao
resultado observado posteriormente.

## 2. Amostras e grupos experimentais

A validacao externa foi configurada para {grupos['total_amostras']} amostras e
{grupos['total_grupos']} grupos `experimento_tratamento`. Esse grupo combina
`experimento_rotulo` e `tratamento_planilha` normalizados, reduzindo o risco de
que amostras do mesmo contexto experimental aparecam simultaneamente em treino
e teste externo.

Menor grupo externo:

{tabela_markdown(grupos['menor_grupo'], ['fold', 'grupo_externo', 'n_teste', 'teste_contaminada', 'teste_nao_contaminada'], 1)}

Maior grupo externo:

{tabela_markdown(grupos['maior_grupo'], ['fold', 'grupo_externo', 'n_teste', 'teste_contaminada', 'teste_nao_contaminada'], 1)}

## 3. Protocolo do split original

O split original usa a divisao treino/validacao/teste consolidada em
`saidas/tabelas/04_dataset_split/divisao_treino_validacao_teste.csv`. Os
modelos e thresholds do split original foram gerados em etapas anteriores; este
script apenas le `comparacao_final_classificacao.csv` e nao recalcula
thresholds.

## 4. Protocolo leave-one-experimento-tratamento-out

Na validacao externa, cada grupo `experimento_tratamento` e deixado de fora uma
vez como teste externo. A validacao interna usa um grupo inteiro do conjunto de
desenvolvimento, escolhido deterministicamente. O split original permanece
apenas como coluna de auditoria.

## 5. Modelos avaliados

Foram consolidados modelos de imagem inteira, YOLO/caixas, ResNet18 com
recortes, Random Forest e SVM com atributos visuais normalizados, MobileNetV2
com recortes, baseline de metadados e baseline sempre-contaminada. O baseline de
metadados e tratado como diagnostico de vies de lote/tratamento, nao como
candidato visual para aplicativo.

## 6. Parametros cientificos principais

```json
{parametros_txt}
```

## 7. Resultados do split original

Resultado com maior balanced accuracy entre linhas oficiais/controles do cenario
equilibrado:

{formatar_linha_resultado(melhor_split_bal)}

Tabela resumida do split original:

{tabela_markdown(split_equilibrado.sort_values(['balanced_accuracy', 'mcc'], ascending=[False, False]), ['modelo', 'cenario', 'conjunto_features', 'balanced_accuracy', 'mcc', 'recall_contaminada', 'especificidade_nao_contaminada', 'f1_contaminada'], 12)}

## 8. Resultados externos micro e macro

Resultado externo micro com maior balanced accuracy entre linhas consolidadas:

{formatar_linha_resultado(melhor_ext_bal)}

Resumo micro:

{tabela_markdown(validacao[validacao['agregacao'].astype(str) == 'micro'].sort_values(['balanced_accuracy', 'mcc'], ascending=[False, False]), ['modelo', 'cenario', 'conjunto_features', 'balanced_accuracy', 'mcc', 'recall_contaminada', 'especificidade_nao_contaminada', 'f1_contaminada', 'folds'], 12)}

Resumo macro:

{tabela_markdown(validacao[validacao['agregacao'].astype(str) == 'macro'].sort_values(['balanced_accuracy_media', 'mcc_media'], ascending=[False, False]), ['modelo', 'cenario', 'conjunto_features', 'balanced_accuracy_media', 'balanced_accuracy_dp', 'mcc_media', 'mcc_dp', 'recall_contaminada_media', 'especificidade_nao_contaminada_media', 'folds'], 12)}

## 9. Comparacao com baseline sempre-contaminada

O baseline sempre-contaminada e um controle obrigatorio: ele tende a maximizar
recall da classe contaminada ao custo de especificidade nula ou muito baixa.
Resultados com F1 alto devem ser interpretados contra esse controle.

Split original:

{tabela_markdown(baseline_split, ['cenario', 'balanced_accuracy', 'mcc', 'recall_contaminada', 'especificidade_nao_contaminada', 'f1_contaminada'], 5)}

Validacao externa:

{tabela_markdown(baseline_ext, ['agregacao', 'cenario', 'balanced_accuracy', 'mcc', 'recall_contaminada', 'especificidade_nao_contaminada', 'f1_contaminada', 'balanced_accuracy_media', 'mcc_media'], 8)}

## 10. Diagnostico de vies de lote/tratamento

O baseline de metadados usa origem, tratamento, pasta e campos derivados. Ele
serve para diagnosticar vies de lote/tratamento e nao deve ser tratado como
modelo visual candidato ao aplicativo.

Split original metadados:

{tabela_markdown(metadados_split, ['cenario', 'balanced_accuracy', 'mcc', 'recall_contaminada', 'especificidade_nao_contaminada', 'f1_contaminada'], 8)}

Validacao externa metadados:

{tabela_markdown(metadados_ext, ['agregacao', 'cenario', 'balanced_accuracy', 'mcc', 'recall_contaminada', 'especificidade_nao_contaminada', 'f1_contaminada', 'balanced_accuracy_media', 'mcc_media'], 8)}

Resumo textual do script 27:

```text
{artefatos['resumo_comparacao']}
```

## 11. Limitacoes

As conclusoes sao limitadas pelo tamanho da base, pelo numero de grupos
experimentais e por possiveis diferencas tecnicas entre lotes, tratamentos,
origens e padroes de imagem. Grupos pequenos reduzem a estabilidade das metricas
por tratamento e ampliam incerteza na leitura macro.

Grupos pequenos no diagnostico dos folds:

{texto_grupos_pequenos}

## 12. Conclusao sobre viabilidade da classificacao

Nao se deve declarar vencedor apenas por F1. A leitura prioritaria deve combinar
balanced accuracy, MCC, recall da contaminada e especificidade da nao
contaminada. Se a validacao externa apresentar queda relevante em balanced
accuracy ou MCC frente ao split original, isso indica fragilidade de
generalizacao e reforca que o problema ainda nao esta pronto para classificacao
direta automatica.

## 13. Justificativa para avancar para triagem preventiva

A classificacao direta exige boa sensibilidade sem destruir a especificidade. O
historico dos experimentos mostra que recall alto pode ser obtido por regras
conservadoras proximas ao baseline sempre-contaminada. Portanto, a etapa
operacional mais defensavel e triagem preventiva: separar alto risco, revisar
casos incertos e evitar liberacao automatica de baixo risco sem validacao
adicional por lote/tratamento.

## Figuras

{figuras_md}

## Arquivos derivados deste relatorio

- `{caminho_relativo(CAMINHO_TABELA_SPLIT)}`
- `{caminho_relativo(CAMINHO_TABELA_VALIDACAO)}`
- `{caminho_relativo(CAMINHO_TABELA_TRATAMENTO)}`
- `{caminho_relativo(CAMINHO_MANIFESTO)}`
"""
    return textwrap.dedent(relatorio).strip() + "\n"


def main():
    print("=" * 70)
    print("GERANDO RELATORIO CIENTIFICO FINAL DA CLASSIFICACAO")
    print("=" * 70)
    print("Este script nao treina modelos e nao recalibra thresholds.")

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
        *figuras,
    ]:
        print(f"- {caminho}")


if __name__ == "__main__":
    main()
