from pathlib import Path
from datetime import datetime
import argparse
import json
import math
import random
import re
import time
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ============================================================
# SCRIPT 28 - VALIDACAO EXTERNA POR TRATAMENTO
# ------------------------------------------------------------
# Objetivo:
# - Testar generalizacao em leave-one-experimento-tratamento-out
# - Manter grupos inteiros fora do treino e da validacao interna
# - Comparar RF, SVM, metadados, MobileNetV2 e controle trivial
#
# Este script pode treinar modelos quando executado sem --preflight.
# O Codex deve validar apenas com compileall.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_DATASET_RECORTADO = PASTA_PROJETO / "saidas" / "dataset_recortado"
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_TABELA_MESTRE = PASTA_TABELAS / "03_tabela_mestre"
PASTA_CLASSICOS = PASTA_TABELAS / "06_modelos" / "classicos"
PASTA_VALIDACAO = PASTA_TABELAS / "07_classificacao_final" / "validacao_tratamento"
PASTA_HISTORICOS_MOBILENET = PASTA_VALIDACAO / "historicos_mobilenetv2"
PASTA_CHECKPOINTS = PASTA_PROJETO / "saidas" / "modelos" / "validacao_tratamento"

CAMINHO_TABELA_MESTRE_PADRAO = PASTA_TABELA_MESTRE / "tabela_mestre.csv"
CAMINHO_TABELA_MESTRE_ALTERNATIVO = PASTA_TABELAS / "tabela_mestre.csv"
CAMINHO_ATRIBUTOS = PASTA_CLASSICOS / "atributos_visuais_recortes.csv"
CAMINHO_FEATURES = PASTA_CLASSICOS / "features_classicos.csv"
CAMINHO_COMPARACAO_SPLIT_ORIGINAL = (
    PASTA_TABELAS / "07_classificacao_final" / "comparacao_final_classificacao.csv"
)

CAMINHO_FOLDS = PASTA_VALIDACAO / "folds_validacao_por_tratamento.csv"
CAMINHO_PREDICOES = PASTA_VALIDACAO / "predicoes_validacao_por_tratamento.csv"
CAMINHO_METRICAS = PASTA_VALIDACAO / "metricas_validacao_por_tratamento.csv"
CAMINHO_THRESHOLDS = PASTA_VALIDACAO / "thresholds_validacao_por_tratamento.csv"
CAMINHO_RESUMO = PASTA_VALIDACAO / "resumo_generalizacao_por_tratamento.csv"
CAMINHO_COMPARACAO_PROTOCOLOS = (
    PASTA_VALIDACAO / "comparacao_split_original_vs_tratamento.csv"
)
CAMINHO_CONFIG = PASTA_VALIDACAO / "config_validacao_por_tratamento.json"
CAMINHO_DIAGNOSTICO_FOLDS = (
    PASTA_VALIDACAO / "diagnostico_folds_validacao_por_tratamento.csv"
)

CLASSES = ["nao_contaminada", "contaminada"]
CLASSE_POSITIVA = "contaminada"
CLASSE_PARA_INDICE = {classe: indice for indice, classe in enumerate(CLASSES)}
INDICE_POSITIVO = CLASSE_PARA_INDICE[CLASSE_POSITIVA]
GRUPO_VALIDACAO_EXTERNA = "experimento_tratamento"
SEMENTE_ALEATORIA = 42
RECALL_MINIMO_PRIORITARIO = 0.95
EPS = 1e-12

CONJUNTO_PRINCIPAL = "principal_normalizado"
CONJUNTO_NAO_APLICAVEL = "nao_aplicavel"
PROTOCOLO = "leave_one_experimento_tratamento_out"

CV_FOLDS_MAX = 5
N_JOBS_GRID = 6

TAMANHO_IMAGEM = 224
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
USAR_MIXED_PRECISION = True
USAR_CHANNELS_LAST_CUDA = True
EPOCHS_TOTAL = 80
EPOCHS_BACKBONE_CONGELADO = 5
PACIENCIA_EARLY_STOPPING = 8
LEARNING_RATE_CLASSIFICADOR = 1e-4
LEARNING_RATE_AJUSTE_FINO = 1e-5
WEIGHT_DECAY = 1e-4
BLOCOS_FINAIS_DESCONGELADOS = 4
PESOS_PRE_TREINADOS = "MobileNet_V2_Weights.DEFAULT"
PESOS_IMAGENET_CARREGADOS = True

MIN_AMOSTRAS_GRUPO_VALIDACAO = 10
MIN_AMOSTRAS_GRUPO_METADADOS = 10
ALPHA_SUAVIZACAO = 10.0

MODELOS_TREINAVEIS = ["random_forest", "svm_rbf", "metadados", "mobilenetv2"]
TODOS_MODELOS = [*MODELOS_TREINAVEIS, "baseline_sempre_contaminada"]

COLUNAS_EXCLUIDAS_OBRIGATORIAS = {
    "nome_arquivo",
    "split",
    "classe",
    "classe_real",
    "alvo",
    "caminho_imagem",
    "caminho_recorte",
    "status_atributos",
    "erro_atributos",
}

COLUNAS_ABSOLUTAS_EXCLUIR_PRINCIPAL = {
    "largura_recorte",
    "altura_recorte",
    "pixels_recorte",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "mask_area_px",
    "contour_area_px",
    "contour_perimeter_px",
}

PREFIXOS_TEXTURA_RESOLUCAO_ORIGINAL = ("texture_", "lbp_")
TERMOS_METADADOS_PROIBIDOS_FEATURES = (
    "origem",
    "tratamento",
    "pasta",
    "experimento",
    "caminho",
    "arquivo",
    "nome",
    "split",
    "classe",
    "alvo",
    "status",
    "erro",
)

COLUNAS_CATEGORICAS_METADADOS = [
    "origem",
    "origem_planilha",
    "experimento_rotulo",
    "experimento_img",
    "tratamento_planilha",
    "tratamento_normalizado",
    "pasta_esperada",
    "pasta_pai",
    "pasta_normalizada",
    "pasta_familia",
    "subpasta_caminho",
    "origem_tratamento",
    "origem_pasta",
    "experimento_tratamento",
    "experimento_pasta",
    "prefixo_id_semente",
    "primeiro_caractere_id",
    "faixa_id_semente",
    "tem_letra_id",
    "tem_numero_id",
    "extensao",
    "modo_cor",
]

COLUNAS_NUMERICAS_METADADOS = [
    "largura",
    "altura",
    "proporcao_imagem",
    "megapixels",
    "qtd_observacoes",
    "numero_id_semente",
    "numero_pasta",
]

COLUNAS_NUMERICAS_BINADAS_METADADOS = [
    f"{coluna}_faixa" for coluna in COLUNAS_NUMERICAS_METADADOS
]
COLUNAS_FEATURES_METADADOS = (
    COLUNAS_CATEGORICAS_METADADOS + COLUNAS_NUMERICAS_BINADAS_METADADOS
)

COLUNAS_METRICAS = [
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
]

COLUNAS_COMPARACAO_PROTOCOLOS = [
    "f1_contaminada",
    "balanced_accuracy",
    "mcc",
    "recall_contaminada",
    "especificidade_nao_contaminada",
]

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


class DatasetRecortes(Dataset):
    def __init__(self, df: pd.DataFrame, transformacao):
        self.df = df.reset_index(drop=True)
        self.transformacao = transformacao

    def __len__(self):
        return len(self.df)

    def __getitem__(self, indice):
        linha = self.df.iloc[indice]
        caminho = PASTA_PROJETO / str(linha["caminho_relativo"])

        with Image.open(caminho) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

        imagem = self.transformacao(img)
        alvo = int(linha["alvo"])
        return imagem, alvo


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Executa validacao externa leave-one-experimento-tratamento-out "
            "para a classificacao."
        )
    )
    parser.add_argument(
        "--modelos",
        nargs="+",
        default=["todos"],
        choices=[*MODELOS_TREINAVEIS, "todos"],
        help="Modelos a executar. Use 'todos' para RF, SVM, metadados e MobileNetV2.",
    )
    parser.add_argument(
        "--somente-grupo",
        default=None,
        help="Executa apenas um grupo externo, pelo valor de experimento_tratamento.",
    )
    parser.add_argument(
        "--retomar",
        action="store_true",
        help="Pula folds/modelos com metricas completas ja salvas.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Valida grupos, juncoes e folds sem treinar modelos.",
    )
    return parser


def modelos_solicitados(valores: list[str]) -> list[str]:
    if "todos" in valores:
        return MODELOS_TREINAVEIS.copy()
    vistos = []
    for valor in valores:
        if valor not in vistos:
            vistos.append(valor)
    return vistos


def nome_seguro(texto: str) -> str:
    saida = str(texto)
    for antigo, novo in {
        "\\": "__",
        "/": "__",
        " ": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
    }.items():
        saida = saida.replace(antigo, novo)
    return saida


def normalizar_texto(valor) -> str:
    if pd.isna(valor):
        return "desconhecido"
    texto = str(valor).strip().lower()
    texto = re.sub(r"\s+", "_", texto)
    texto = re.sub(r"[^a-z0-9_]+", "_", texto)
    texto = re.sub(r"_+", "_", texto).strip("_")
    return texto or "desconhecido"


def localizar_tabela_mestre() -> Path:
    if CAMINHO_TABELA_MESTRE_PADRAO.exists():
        return CAMINHO_TABELA_MESTRE_PADRAO
    if CAMINHO_TABELA_MESTRE_ALTERNATIVO.exists():
        return CAMINHO_TABELA_MESTRE_ALTERNATIVO
    raise FileNotFoundError(
        "Tabela mestre nao encontrada em "
        f"{CAMINHO_TABELA_MESTRE_PADRAO} nem em {CAMINHO_TABELA_MESTRE_ALTERNATIVO}"
    )


def extrair_partes_id(valor) -> dict:
    texto = "" if pd.isna(valor) else str(valor).strip().lower()
    match_prefixo = re.match(r"^[a-z]+", texto)
    match_numero = re.search(r"\d+", texto)

    numero = np.nan
    if match_numero:
        numero = float(match_numero.group(0))

    if np.isnan(numero):
        faixa = "sem_numero"
    elif numero <= 5:
        faixa = "001_005"
    elif numero <= 10:
        faixa = "006_010"
    elif numero <= 20:
        faixa = "011_020"
    elif numero <= 40:
        faixa = "021_040"
    else:
        faixa = "041_mais"

    return {
        "prefixo_id_semente": match_prefixo.group(0) if match_prefixo else "sem_prefixo",
        "primeiro_caractere_id": texto[:1] if texto else "vazio",
        "numero_id_semente": numero,
        "faixa_id_semente": faixa,
        "tem_letra_id": "sim" if re.search(r"[a-z]", texto) else "nao",
        "tem_numero_id": "sim" if re.search(r"\d", texto) else "nao",
    }


def extrair_numero_pasta(valor) -> float:
    texto = "" if pd.isna(valor) else str(valor).lower()
    numeros = re.findall(r"\d+", texto)
    if not numeros:
        return np.nan
    return float(numeros[-1])


def criar_faixa_numerica(serie: pd.Series) -> pd.Series:
    valores = pd.to_numeric(serie, errors="coerce")
    if valores.notna().sum() == 0:
        return pd.Series(["sem_valor"] * len(serie), index=serie.index)

    if valores.dropna().nunique() <= 8:
        return valores.map(lambda x: "sem_valor" if pd.isna(x) else f"valor_{x:g}")

    try:
        faixas = pd.qcut(valores, q=5, duplicates="drop")
        return faixas.astype(str).replace("nan", "sem_valor")
    except ValueError:
        return valores.map(lambda x: "sem_valor" if pd.isna(x) else f"valor_{x:g}")


def carregar_atributos() -> pd.DataFrame:
    if not CAMINHO_ATRIBUTOS.exists():
        raise FileNotFoundError(
            f"Atributos visuais nao encontrados: {CAMINHO_ATRIBUTOS}\n"
            "Execute antes o script 22."
        )

    df = pd.read_csv(CAMINHO_ATRIBUTOS)
    obrigatorias = ["nome_arquivo", "classe", "alvo"]
    faltantes = [coluna for coluna in obrigatorias if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas obrigatorias ausentes nos atributos: {faltantes}")

    if "status_atributos" in df.columns:
        erros = df[df["status_atributos"].astype(str) != "ok"].copy()
        if not erros.empty:
            raise ValueError(
                "Ha registros sem atributos visuais validos em "
                f"{CAMINHO_ATRIBUTOS}. Corrija antes da validacao externa."
            )

    if df["nome_arquivo"].duplicated().any():
        duplicados = df[df["nome_arquivo"].duplicated(keep=False)]["nome_arquivo"].tolist()
        raise ValueError(f"Nomes de arquivo duplicados nos atributos: {duplicados[:20]}")

    df = df.copy()
    df["classe"] = df["classe"].astype(str)
    df = df[df["classe"].isin(CLASSES)].copy()
    df["alvo"] = pd.to_numeric(df["alvo"], errors="raise").astype(int)
    df["split_original"] = df["split"].astype(str) if "split" in df.columns else "nao_informado"

    if "caminho_recorte" not in df.columns:
        df["caminho_recorte"] = [
            str((PASTA_DATASET_RECORTADO / classe / nome).relative_to(PASTA_PROJETO))
            for classe, nome in zip(df["classe"], df["nome_arquivo"])
        ]
    df["caminho_relativo"] = df["caminho_recorte"].astype(str)
    return df.reset_index(drop=True)


def adicionar_features_metadados(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base["classe_real"] = base["classe"]
    base["alvo_real"] = base["alvo"].astype(int)

    if "origem_planilha" in base.columns:
        base["origem"] = base["origem_planilha"]
    elif "origem" not in base.columns:
        base["origem"] = "desconhecido"

    caminho_relativo = base.get("caminho_relativo_original", base.get("caminho_relativo", ""))
    partes = (
        pd.Series(caminho_relativo, index=base.index)
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.split("/", expand=True)
    )
    base["experimento_caminho"] = partes[0].fillna("desconhecido")
    base["subpasta_caminho"] = partes[1].fillna("desconhecido") if partes.shape[1] > 1 else "desconhecido"

    for coluna in ["experimento_rotulo", "tratamento_planilha", "pasta_esperada"]:
        if coluna not in base.columns:
            base[coluna] = "desconhecido"

    base["tratamento_normalizado"] = base["tratamento_planilha"].map(normalizar_texto)
    base["pasta_normalizada"] = base["pasta_esperada"].map(normalizar_texto)
    base["pasta_familia"] = (
        base["pasta_esperada"]
        .map(normalizar_texto)
        .str.replace(r"_?\d+$", "", regex=True)
        .replace("", "desconhecido")
    )
    base["experimento_tratamento"] = (
        base["experimento_rotulo"].map(normalizar_texto)
        + "__"
        + base["tratamento_planilha"].map(normalizar_texto)
    )
    base["origem_tratamento"] = (
        base["origem"].map(normalizar_texto)
        + "__"
        + base["tratamento_planilha"].map(normalizar_texto)
    )
    base["origem_pasta"] = (
        base["origem"].map(normalizar_texto)
        + "__"
        + base["pasta_esperada"].map(normalizar_texto)
    )
    base["experimento_pasta"] = (
        base["experimento_rotulo"].map(normalizar_texto)
        + "__"
        + base["pasta_esperada"].map(normalizar_texto)
    )

    id_busca = base["id_busca"] if "id_busca" in base.columns else base["nome_arquivo"]
    partes_id = id_busca.apply(extrair_partes_id).apply(pd.Series)
    base = pd.concat([base, partes_id], axis=1)
    base["numero_pasta"] = base["pasta_esperada"].apply(extrair_numero_pasta)

    for coluna in ["largura", "altura", "qtd_observacoes"]:
        if coluna not in base.columns:
            base[coluna] = np.nan
        base[coluna] = pd.to_numeric(base[coluna], errors="coerce")

    base["proporcao_imagem"] = base["largura"] / base["altura"].replace(0, np.nan)
    base["megapixels"] = (base["largura"] * base["altura"]) / 1_000_000

    for coluna in COLUNAS_CATEGORICAS_METADADOS:
        if coluna not in base.columns:
            base[coluna] = "desconhecido"
        base[coluna] = base[coluna].fillna("desconhecido").astype(str)

    for coluna in COLUNAS_NUMERICAS_METADADOS:
        if coluna not in base.columns:
            base[coluna] = np.nan
        base[coluna] = pd.to_numeric(base[coluna], errors="coerce")

    for coluna in COLUNAS_NUMERICAS_METADADOS:
        base[f"{coluna}_faixa"] = (
            criar_faixa_numerica(base[coluna])
            .fillna("sem_valor")
            .astype(str)
        )

    return base


def carregar_base_experimento() -> pd.DataFrame:
    atributos = carregar_atributos()
    caminho_tabela = localizar_tabela_mestre()
    tabela = pd.read_csv(caminho_tabela)

    if "caminho_relativo" not in tabela.columns:
        raise ValueError("A tabela mestre precisa conter a coluna caminho_relativo.")

    tabela = tabela.copy()
    if "status" in tabela.columns:
        tabela = tabela[tabela["status"].astype(str) == "ok"].copy()
    if "imagem_valida" in tabela.columns:
        tabela = tabela[tabela["imagem_valida"].astype(str).str.lower() == "true"].copy()
    if "classe" in tabela.columns:
        tabela = tabela[tabela["classe"].isin(CLASSES)].copy()

    tabela["nome_copiado"] = tabela["caminho_relativo"].map(nome_seguro)
    if tabela["nome_copiado"].duplicated().any():
        duplicados = tabela[tabela["nome_copiado"].duplicated(keep=False)]["nome_copiado"].tolist()
        raise ValueError(f"Nomes copiados duplicados na tabela mestre: {duplicados[:20]}")

    tabela = tabela.rename(columns={"caminho_relativo": "caminho_relativo_original"})
    base = atributos.merge(
        tabela,
        left_on="nome_arquivo",
        right_on="nome_copiado",
        how="left",
        suffixes=("", "_metadados"),
        validate="one_to_one",
    )

    sem_metadados = base[base["nome_copiado"].isna()].copy()
    if not sem_metadados.empty:
        raise ValueError(
            "Ha registros dos atributos sem metadados na tabela mestre. Exemplos: "
            f"{sem_metadados['nome_arquivo'].head(20).tolist()}"
        )

    base = adicionar_features_metadados(base)
    base = base.sort_values("nome_arquivo").reset_index(drop=True)

    if base[GRUPO_VALIDACAO_EXTERNA].isna().any():
        raise ValueError(f"Ha valores ausentes em {GRUPO_VALIDACAO_EXTERNA}.")
    if base["nome_arquivo"].duplicated().any():
        raise ValueError("A base consolidada contem nome_arquivo duplicado.")

    return base


def carregar_features_principais(df: pd.DataFrame) -> list[str]:
    if not CAMINHO_FEATURES.exists():
        raise FileNotFoundError(
            f"Arquivo de features nao encontrado: {CAMINHO_FEATURES}\n"
            "Execute antes o script 23 para registrar o conjunto principal."
        )

    features_df = pd.read_csv(CAMINHO_FEATURES)
    if "conjunto_features" not in features_df.columns or "feature" not in features_df.columns:
        raise ValueError(
            f"{CAMINHO_FEATURES} precisa conter as colunas conjunto_features e feature."
        )

    features = (
        features_df[features_df["conjunto_features"].astype(str) == CONJUNTO_PRINCIPAL]["feature"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if not features:
        raise ValueError(f"Nenhuma feature encontrada para {CONJUNTO_PRINCIPAL}.")

    faltantes = [feature for feature in features if feature not in df.columns]
    if faltantes:
        raise ValueError(f"Features do conjunto principal ausentes na base: {faltantes[:30]}")

    proibidas = []
    for feature in features:
        if feature in COLUNAS_EXCLUIDAS_OBRIGATORIAS:
            proibidas.append(feature)
        if feature in COLUNAS_ABSOLUTAS_EXCLUIR_PRINCIPAL:
            proibidas.append(feature)
        if feature.startswith(PREFIXOS_TEXTURA_RESOLUCAO_ORIGINAL):
            proibidas.append(feature)
        if any(termo in feature.lower() for termo in TERMOS_METADADOS_PROIBIDOS_FEATURES):
            proibidas.append(feature)

    if proibidas:
        raise ValueError(
            "O conjunto principal contem features proibidas para a validacao visual: "
            f"{sorted(set(proibidas))}"
        )

    return sorted(features)


def preparar_matriz(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    matriz = df[features].apply(pd.to_numeric, errors="coerce")
    return matriz.replace([np.inf, -np.inf], np.nan)


def contagem_classes(df: pd.DataFrame) -> dict:
    contagens = df["alvo"].value_counts().reindex([0, 1], fill_value=0).astype(int)
    return {
        "nao_contaminada": int(contagens.loc[0]),
        "contaminada": int(contagens.loc[1]),
    }


def calcular_n_folds_cv(df_treino: pd.DataFrame) -> int:
    contagens_classe = df_treino["alvo"].value_counts().reindex([0, 1], fill_value=0)
    grupos_unicos = df_treino[GRUPO_VALIDACAO_EXTERNA].nunique()
    return int(min(CV_FOLDS_MAX, grupos_unicos, int(contagens_classe.min())))


def escolher_grupo_validacao(df_desenvolvimento: pd.DataFrame) -> str | None:
    taxa_desenvolvimento = float(df_desenvolvimento["alvo"].mean())
    candidatos = []
    for grupo, df_grupo in df_desenvolvimento.groupby(GRUPO_VALIDACAO_EXTERNA):
        contagens = df_grupo["alvo"].value_counts().reindex([0, 1], fill_value=0)
        possui_duas_classes = bool((contagens > 0).all())
        total = int(len(df_grupo))
        if not possui_duas_classes or total < MIN_AMOSTRAS_GRUPO_VALIDACAO:
            continue
        taxa = float(df_grupo["alvo"].mean())
        candidatos.append({
            "grupo": str(grupo),
            "total": total,
            "taxa_contaminacao": taxa,
            "distancia_taxa": abs(taxa - taxa_desenvolvimento),
        })

    if not candidatos:
        return None

    candidatos_df = pd.DataFrame(candidatos).sort_values(
        ["distancia_taxa", "grupo"],
        ascending=[True, True],
    )
    return str(candidatos_df.iloc[0]["grupo"])


def criar_folds(base: pd.DataFrame, somente_grupo: str | None) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    grupos = sorted(base[GRUPO_VALIDACAO_EXTERNA].astype(str).unique().tolist())
    if somente_grupo:
        grupo_normalizado = normalizar_texto(somente_grupo)
        if somente_grupo in grupos:
            grupos = [somente_grupo]
        elif grupo_normalizado in grupos:
            grupos = [grupo_normalizado]
        else:
            raise ValueError(
                f"Grupo solicitado nao encontrado: {somente_grupo}. "
                f"Exemplos disponiveis: {grupos[:20]}"
            )

    folds = []
    diagnosticos = []
    registros_folds = []

    for fold_id, grupo_externo in enumerate(grupos, start=1):
        teste = base[base[GRUPO_VALIDACAO_EXTERNA] == grupo_externo].copy()
        desenvolvimento = base[base[GRUPO_VALIDACAO_EXTERNA] != grupo_externo].copy()
        grupo_validacao = escolher_grupo_validacao(desenvolvimento)

        if grupo_validacao is None:
            validacao = pd.DataFrame(columns=base.columns)
            treino = desenvolvimento.copy()
        else:
            validacao = desenvolvimento[
                desenvolvimento[GRUPO_VALIDACAO_EXTERNA] == grupo_validacao
            ].copy()
            treino = desenvolvimento[
                desenvolvimento[GRUPO_VALIDACAO_EXTERNA] != grupo_validacao
            ].copy()

        folds.append({
            "fold": fold_id,
            "grupo_externo": grupo_externo,
            "grupo_validacao": grupo_validacao,
            "indices_treino": treino.index.tolist(),
            "indices_validacao": validacao.index.tolist(),
            "indices_teste": teste.index.tolist(),
        })

        problemas = []
        for nome_papel, df_papel in [
            ("treino", treino),
            ("validacao", validacao),
            ("teste", teste),
        ]:
            contagens = df_papel["alvo"].value_counts().reindex([0, 1], fill_value=0)
            if not (contagens > 0).all():
                problemas.append(f"{nome_papel}_sem_duas_classes")
            if df_papel["nome_arquivo"].duplicated().any():
                problemas.append(f"{nome_papel}_com_nome_arquivo_duplicado")

        conjuntos = {
            "treino": set(treino["nome_arquivo"]),
            "validacao": set(validacao["nome_arquivo"]),
            "teste": set(teste["nome_arquivo"]),
        }
        if conjuntos["treino"] & conjuntos["validacao"]:
            problemas.append("sobreposicao_treino_validacao")
        if conjuntos["treino"] & conjuntos["teste"]:
            problemas.append("sobreposicao_treino_teste")
        if conjuntos["validacao"] & conjuntos["teste"]:
            problemas.append("sobreposicao_validacao_teste")

        n_cv = calcular_n_folds_cv(treino)
        if n_cv < 2:
            problemas.append("cv_classicos_menos_de_2_folds")

        diagnostico = {
            "fold": fold_id,
            "grupo_externo": grupo_externo,
            "grupo_validacao": grupo_validacao,
            "valido": len(problemas) == 0,
            "problemas": ";".join(problemas),
            "n_treino": int(len(treino)),
            "n_validacao": int(len(validacao)),
            "n_teste": int(len(teste)),
            "cv_folds_classicos": n_cv,
            **{f"treino_{k}": v for k, v in contagem_classes(treino).items()},
            **{f"validacao_{k}": v for k, v in contagem_classes(validacao).items()},
            **{f"teste_{k}": v for k, v in contagem_classes(teste).items()},
        }
        diagnosticos.append(diagnostico)

        for papel, df_papel in [
            ("treino_interno", treino),
            ("validacao_interna", validacao),
            ("teste_externo", teste),
        ]:
            for _, linha in df_papel.iterrows():
                registros_folds.append({
                    "fold": fold_id,
                    "grupo_externo": grupo_externo,
                    "grupo_validacao": grupo_validacao,
                    "papel_amostra": papel,
                    "nome_arquivo": linha["nome_arquivo"],
                    "caminho_relativo": linha["caminho_relativo"],
                    "classe": linha["classe"],
                    "alvo": int(linha["alvo"]),
                    "split_original": linha["split_original"],
                    GRUPO_VALIDACAO_EXTERNA: linha[GRUPO_VALIDACAO_EXTERNA],
                })

    diagnosticos_df = pd.DataFrame(diagnosticos)
    folds_df = pd.DataFrame(registros_folds)
    validar_cobertura_teste_externo(base, folds_df, somente_grupo is not None)
    return folds, folds_df, diagnosticos_df


def validar_cobertura_teste_externo(base: pd.DataFrame, folds_df: pd.DataFrame, parcial: bool):
    teste = folds_df[folds_df["papel_amostra"] == "teste_externo"].copy()
    contagens = teste["nome_arquivo"].value_counts()
    duplicados = contagens[contagens != 1]
    if not duplicados.empty:
        raise ValueError(
            "Cada amostra deve aparecer exatamente uma vez como teste externo. "
            f"Problemas: {duplicados.head(20).to_dict()}"
        )

    if not parcial:
        esperados = set(base["nome_arquivo"])
        encontrados = set(teste["nome_arquivo"])
        faltantes = sorted(esperados - encontrados)
        extras = sorted(encontrados - esperados)
        if faltantes or extras:
            raise ValueError(
                "Cobertura externa invalida. "
                f"faltantes={faltantes[:20]} extras={extras[:20]}"
            )


def validar_folds_antes_do_treino(diagnosticos_df: pd.DataFrame):
    invalidos = diagnosticos_df[~diagnosticos_df["valido"].astype(bool)]
    if not invalidos.empty:
        PASTA_VALIDACAO.mkdir(parents=True, exist_ok=True)
        diagnosticos_df.to_csv(CAMINHO_DIAGNOSTICO_FOLDS, index=False, encoding="utf-8-sig")
        raise ValueError(
            "Folds invalidos detectados antes do treino. "
            f"Diagnostico salvo em: {CAMINHO_DIAGNOSTICO_FOLDS}"
        )


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
        "total": int(total),
        "suporte_contaminada": int(suporte_contaminada),
        "suporte_nao_contaminada": int(suporte_nao_contaminada),
    }


def calcular_metricas_probabilidade(y_real, prob_contaminada, threshold) -> dict:
    y_real = np.asarray(y_real, dtype=int)
    prob_contaminada = np.asarray(prob_contaminada, dtype=float)
    pred = (prob_contaminada >= float(threshold)).astype(int)
    tn = int(((y_real == 0) & (pred == 0)).sum())
    fp = int(((y_real == 0) & (pred == 1)).sum())
    fn = int(((y_real == 1) & (pred == 0)).sum())
    tp = int(((y_real == 1) & (pred == 1)).sum())
    return {
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        **calcular_metricas_confusao(tn, fp, fn, tp),
    }


def gerar_curva_threshold(y_validacao, prob_validacao) -> pd.DataFrame:
    registros = []
    for threshold in np.arange(0.01, 1.00, 0.01):
        registros.append(
            calcular_metricas_probabilidade(
                y_validacao,
                prob_validacao,
                round(float(threshold), 2),
            )
        )
    return pd.DataFrame(registros)


def escolher_threshold_por_f1(df_thresholds: pd.DataFrame) -> float:
    melhor = df_thresholds.sort_values(
        ["f1_contaminada", "recall_contaminada", "especificidade_nao_contaminada"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(melhor["threshold"])


def escolher_threshold_por_recall(df_thresholds: pd.DataFrame) -> float:
    candidatos = df_thresholds[
        df_thresholds["recall_contaminada"] >= RECALL_MINIMO_PRIORITARIO
    ].copy()
    if candidatos.empty:
        candidatos = df_thresholds.copy()
        ordenacao = ["recall_contaminada", "f1_contaminada", "especificidade_nao_contaminada"]
    else:
        ordenacao = ["f1_contaminada", "especificidade_nao_contaminada", "recall_contaminada"]
    melhor = candidatos.sort_values(ordenacao, ascending=[False, False, False]).iloc[0]
    return float(melhor["threshold"])


def criar_scoring_classicos() -> dict:
    return {
        "f1_contaminada": make_scorer(f1_score, pos_label=INDICE_POSITIVO, zero_division=0),
        "recall_contaminada": make_scorer(recall_score, pos_label=INDICE_POSITIVO, zero_division=0),
        "precisao_contaminada": make_scorer(
            precision_score,
            pos_label=INDICE_POSITIVO,
            zero_division=0,
        ),
        "especificidade_nao_contaminada": make_scorer(recall_score, pos_label=0, zero_division=0),
        "acuracia": make_scorer(accuracy_score),
    }


def selecionar_melhor_cv(cv_resultados: pd.DataFrame) -> pd.Series:
    return cv_resultados.sort_values(
        [
            "mean_test_f1_contaminada",
            "mean_test_recall_contaminada",
            "mean_test_especificidade_nao_contaminada",
            "mean_test_acuracia",
            "std_test_f1_contaminada",
        ],
        ascending=[False, False, False, False, True],
    ).iloc[0]


def criar_estimador_classico(nome_modelo: str):
    if nome_modelo == "random_forest":
        estimador = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "modelo",
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    random_state=SEMENTE_ALEATORIA,
                    n_jobs=1,
                ),
            ),
        ])
        grid = {
            "modelo__n_estimators": [500, 1000],
            "modelo__max_depth": [None, 8, 16],
            "modelo__min_samples_leaf": [1, 3, 5],
            "modelo__min_samples_split": [2, 5, 10],
            "modelo__max_features": ["sqrt", "log2"],
        }
        return estimador, grid

    if nome_modelo == "svm_rbf":
        estimador = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "modelo",
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=SEMENTE_ALEATORIA,
                ),
            ),
        ])
        grid = {
            "modelo__C": [0.1, 1, 3, 10, 30, 100],
            "modelo__gamma": ["scale", 0.03, 0.01, 0.003, 0.001],
        }
        return estimador, grid

    raise ValueError(f"Modelo classico desconhecido: {nome_modelo}")


def contexto_modelo(nome_modelo: str) -> dict:
    if nome_modelo == "random_forest":
        return {
            "modelo": "random_forest",
            "familia_modelo": "random_forest",
            "tipo_entrada": "atributos_visuais_recortes",
            "usa_pixels": False,
            "usa_recorte": True,
            "usa_atributos_visuais": True,
            "usa_metadados": False,
            "conjunto_features": CONJUNTO_PRINCIPAL,
            "resultado_oficial": True,
            "papel_experimento": "modelo_visual_classico",
        }
    if nome_modelo == "svm_rbf":
        return {
            "modelo": "svm_rbf",
            "familia_modelo": "svm_rbf",
            "tipo_entrada": "atributos_visuais_recortes",
            "usa_pixels": False,
            "usa_recorte": True,
            "usa_atributos_visuais": True,
            "usa_metadados": False,
            "conjunto_features": CONJUNTO_PRINCIPAL,
            "resultado_oficial": True,
            "papel_experimento": "diagnostico_comparativo",
        }
    if nome_modelo == "metadados":
        return {
            "modelo": "metadados_taxas_suavizadas",
            "familia_modelo": "baseline_metadados",
            "tipo_entrada": "metadados",
            "usa_pixels": False,
            "usa_recorte": False,
            "usa_atributos_visuais": False,
            "usa_metadados": True,
            "conjunto_features": CONJUNTO_NAO_APLICAVEL,
            "resultado_oficial": False,
            "papel_experimento": "diagnostico_vies",
        }
    if nome_modelo == "mobilenetv2":
        return {
            "modelo": "mobilenetv2_recortes",
            "familia_modelo": "cnn_mobilenetv2",
            "tipo_entrada": "recorte",
            "usa_pixels": True,
            "usa_recorte": True,
            "usa_atributos_visuais": False,
            "usa_metadados": False,
            "conjunto_features": CONJUNTO_NAO_APLICAVEL,
            "resultado_oficial": True,
            "papel_experimento": "modelo_visual",
        }
    if nome_modelo == "baseline_sempre_contaminada":
        return {
            "modelo": "baseline_sempre_contaminada",
            "familia_modelo": "controle",
            "tipo_entrada": "controle",
            "usa_pixels": False,
            "usa_recorte": False,
            "usa_atributos_visuais": False,
            "usa_metadados": False,
            "conjunto_features": CONJUNTO_NAO_APLICAVEL,
            "resultado_oficial": False,
            "papel_experimento": "controle",
        }
    raise ValueError(f"Contexto desconhecido: {nome_modelo}")


def completar_contexto_fold(contexto: dict, fold: dict) -> dict:
    return {
        **contexto,
        "protocolo": PROTOCOLO,
        "fold": int(fold["fold"]),
        "grupo_externo": fold["grupo_externo"],
        "grupo_validacao": fold["grupo_validacao"],
        "seed": SEMENTE_ALEATORIA,
    }


def avaliar_probabilidades(
    df_validacao: pd.DataFrame,
    df_teste: pd.DataFrame,
    prob_validacao: np.ndarray,
    prob_teste: np.ndarray,
    contexto: dict,
    tempo_treino_segundos: float,
    melhor_epoca=None,
    melhor_loss_validacao=None,
    parametros_json: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_validacao = df_validacao["alvo"].to_numpy(dtype=int)
    y_teste = df_teste["alvo"].to_numpy(dtype=int)

    thresholds = gerar_curva_threshold(y_validacao, prob_validacao)
    threshold_f1 = escolher_threshold_por_f1(thresholds)
    threshold_recall = escolher_threshold_por_recall(thresholds)

    thresholds = thresholds.copy()
    for chave, valor in contexto.items():
        thresholds[chave] = valor
    thresholds["selecionado_melhor_f1_validacao"] = thresholds["threshold"].eq(threshold_f1)
    thresholds["selecionado_prioridade_recall_validacao"] = thresholds["threshold"].eq(threshold_recall)

    cenarios = [
        ("teste_threshold_0_50", 0.50),
        ("teste_threshold_melhor_f1_validacao", threshold_f1),
        ("teste_threshold_prioridade_recall_validacao", threshold_recall),
    ]

    metricas = []
    predicoes = []
    for cenario, threshold in cenarios:
        metricas_cenario = calcular_metricas_probabilidade(y_teste, prob_teste, threshold)
        metricas.append({
            **contexto,
            "cenario": cenario,
            **metricas_cenario,
            "tempo_treino_segundos": round(float(tempo_treino_segundos), 3),
            "melhor_epoca": melhor_epoca,
            "melhor_loss_validacao": melhor_loss_validacao,
            "parametros_json": parametros_json,
        })

        pred_bin = (prob_teste >= float(threshold)).astype(int)
        for indice, (_, linha) in enumerate(df_teste.iterrows()):
            predicoes.append({
                **contexto,
                "cenario": cenario,
                "threshold": float(threshold),
                "nome_arquivo": linha["nome_arquivo"],
                "caminho_relativo": linha["caminho_relativo"],
                "classe_real": linha["classe"],
                "alvo": int(linha["alvo"]),
                "prob_contaminada": float(prob_teste[indice]),
                "predicao": CLASSES[int(pred_bin[indice])],
                "papel_amostra": "teste_externo",
                "split_original": linha["split_original"],
            })

    return pd.DataFrame(metricas), pd.DataFrame(predicoes), thresholds


def treinar_classico_fold(
    base: pd.DataFrame,
    features: list[str],
    fold: dict,
    nome_modelo: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_treino = base.loc[fold["indices_treino"]].copy()
    df_validacao = base.loc[fold["indices_validacao"]].copy()
    df_teste = base.loc[fold["indices_teste"]].copy()

    x_treino = preparar_matriz(df_treino, features)
    y_treino = df_treino["alvo"].to_numpy(dtype=int)
    grupos_treino = df_treino[GRUPO_VALIDACAO_EXTERNA].astype(str).to_numpy()
    x_validacao = preparar_matriz(df_validacao, features)
    x_teste = preparar_matriz(df_teste, features)

    estimador, grid = criar_estimador_classico(nome_modelo)
    n_folds = calcular_n_folds_cv(df_treino)
    if n_folds < 2:
        raise ValueError(
            f"Fold {fold['fold']} nao permite CV estratificada por grupo para {nome_modelo}."
        )

    inicio = time.time()
    cv = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=SEMENTE_ALEATORIA,
    )
    busca = GridSearchCV(
        estimator=estimador,
        param_grid=grid,
        scoring=criar_scoring_classicos(),
        refit=False,
        cv=cv,
        n_jobs=N_JOBS_GRID,
        pre_dispatch=N_JOBS_GRID,
        verbose=0,
        return_train_score=False,
    )
    busca.fit(x_treino, y_treino, groups=grupos_treino)
    cv_resultados = pd.DataFrame(busca.cv_results_)
    melhor = selecionar_melhor_cv(cv_resultados)
    melhores_parametros = dict(melhor["params"])

    modelo = clone(estimador)
    modelo.set_params(**melhores_parametros)
    modelo.fit(x_treino, y_treino)
    tempo_treino = time.time() - inicio

    prob_validacao = modelo.predict_proba(x_validacao)[:, INDICE_POSITIVO]
    prob_teste = modelo.predict_proba(x_teste)[:, INDICE_POSITIVO]
    parametros_json = json.dumps(
        {
            "cv_folds": n_folds,
            "melhores_parametros": {
                chave.replace("modelo__", ""): valor
                for chave, valor in melhores_parametros.items()
            },
            "mean_cv_f1_contaminada": float(melhor["mean_test_f1_contaminada"]),
            "mean_cv_recall_contaminada": float(melhor["mean_test_recall_contaminada"]),
            "mean_cv_especificidade_nao_contaminada": float(
                melhor["mean_test_especificidade_nao_contaminada"]
            ),
        },
        sort_keys=True,
    )

    contexto = completar_contexto_fold(contexto_modelo(nome_modelo), fold)
    return avaliar_probabilidades(
        df_validacao,
        df_teste,
        prob_validacao,
        prob_teste,
        contexto,
        tempo_treino,
        parametros_json=parametros_json,
    )


def logit(probabilidade) -> np.ndarray:
    p = np.clip(probabilidade, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(valor) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(valor)))


def treinar_modelo_metadados(treino: pd.DataFrame) -> dict:
    taxa_global = float(treino["alvo"].mean())
    estatisticas = []

    for coluna in COLUNAS_FEATURES_METADADOS:
        resumo = (
            treino.groupby(coluna, dropna=False)["alvo"]
            .agg(total="size", contaminadas="sum")
            .reset_index()
        )
        resumo["taxa_observada"] = resumo["contaminadas"] / resumo["total"].replace(0, np.nan)
        resumo["taxa_suavizada"] = (
            resumo["contaminadas"] + ALPHA_SUAVIZACAO * taxa_global
        ) / (resumo["total"] + ALPHA_SUAVIZACAO)

        grupos_validos = resumo[resumo["total"] >= MIN_AMOSTRAS_GRUPO_METADADOS].copy()
        if len(grupos_validos) >= 2:
            taxa_minima = float(grupos_validos["taxa_observada"].min())
            taxa_maxima = float(grupos_validos["taxa_observada"].max())
        else:
            taxa_minima = float(resumo["taxa_observada"].min())
            taxa_maxima = float(resumo["taxa_observada"].max())

        amplitude = taxa_maxima - taxa_minima
        estatisticas.append({
            "campo": coluna,
            "mapa_taxa": dict(zip(resumo[coluna].astype(str), resumo["taxa_suavizada"])),
            "peso": max(float(amplitude), 0.01),
            "taxa_global": taxa_global,
        })

    return {"taxa_global": taxa_global, "estatisticas": estatisticas}


def predizer_modelo_metadados(modelo: dict, df: pd.DataFrame) -> np.ndarray:
    soma_logits = np.zeros(len(df), dtype=float)
    soma_pesos = 0.0

    for estatistica in modelo["estatisticas"]:
        coluna = estatistica["campo"]
        peso = float(estatistica["peso"])
        taxas = (
            df[coluna]
            .astype(str)
            .map(estatistica["mapa_taxa"])
            .fillna(estatistica["taxa_global"])
            .astype(float)
            .to_numpy()
        )
        soma_logits += peso * logit(taxas)
        soma_pesos += peso

    if soma_pesos <= 0:
        return np.full(len(df), modelo["taxa_global"], dtype=float)
    return sigmoid(soma_logits / soma_pesos)


def treinar_metadados_fold(
    base: pd.DataFrame,
    fold: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_treino = base.loc[fold["indices_treino"]].copy()
    df_validacao = base.loc[fold["indices_validacao"]].copy()
    df_teste = base.loc[fold["indices_teste"]].copy()

    inicio = time.time()
    modelo = treinar_modelo_metadados(df_treino)
    tempo_treino = time.time() - inicio

    prob_validacao = predizer_modelo_metadados(modelo, df_validacao)
    prob_teste = predizer_modelo_metadados(modelo, df_teste)

    contexto = completar_contexto_fold(contexto_modelo("metadados"), fold)
    return avaliar_probabilidades(
        df_validacao,
        df_teste,
        prob_validacao,
        prob_teste,
        contexto,
        tempo_treino,
        parametros_json=json.dumps(
            {
                "alpha_suavizacao": ALPHA_SUAVIZACAO,
                "min_amostras_grupo": MIN_AMOSTRAS_GRUPO_METADADOS,
                "features": COLUNAS_FEATURES_METADADOS,
            },
            sort_keys=True,
        ),
    )


def configurar_semente_torch(seed: int = SEMENTE_ALEATORIA):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def criar_transformacoes_mobilenet():
    media = [0.485, 0.456, 0.406]
    desvio = [0.229, 0.224, 0.225]

    transformacao_treino = transforms.Compose([
        transforms.RandomResizedCrop(TAMANHO_IMAGEM, scale=(0.80, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.15,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=media, std=desvio),
    ])

    transformacao_validacao = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(TAMANHO_IMAGEM),
        transforms.ToTensor(),
        transforms.Normalize(mean=media, std=desvio),
    ])

    return transformacao_treino, transformacao_validacao


def inicializar_worker_mobilenet(worker_id: int):
    seed = SEMENTE_ALEATORIA + worker_id
    random.seed(seed)
    np.random.seed(seed)


def criar_data_loader_mobilenet(dataset, shuffle: bool) -> DataLoader:
    usar_workers = NUM_WORKERS > 0
    gerador = torch.Generator()
    gerador.manual_seed(SEMENTE_ALEATORIA)
    opcoes = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS and usar_workers,
        "worker_init_fn": inicializar_worker_mobilenet if usar_workers else None,
        "generator": gerador,
    }
    if usar_workers:
        opcoes["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(dataset, **opcoes)


def criar_modelo_mobilenetv2():
    try:
        pesos = models.MobileNet_V2_Weights.DEFAULT
        modelo = models.mobilenet_v2(weights=pesos)
    except Exception as erro:
        raise RuntimeError(
            "Nao foi possivel carregar MobileNet_V2_Weights.DEFAULT. "
            "A validacao externa nao permite MobileNetV2 com weights=None."
        ) from erro

    entrada = modelo.classifier[1].in_features
    modelo.classifier[1] = nn.Linear(entrada, len(CLASSES))
    return modelo


def congelar_backbone(modelo):
    for parametro in modelo.features.parameters():
        parametro.requires_grad = False
    for parametro in modelo.classifier.parameters():
        parametro.requires_grad = True


def liberar_ultimos_blocos(modelo):
    for parametro in modelo.features.parameters():
        parametro.requires_grad = False
    blocos = list(modelo.features.children())[-BLOCOS_FINAIS_DESCONGELADOS:]
    for bloco in blocos:
        for parametro in bloco.parameters():
            parametro.requires_grad = True
    for parametro in modelo.classifier.parameters():
        parametro.requires_grad = True


def parametros_treinaveis(modelo):
    return [parametro for parametro in modelo.parameters() if parametro.requires_grad]


def criar_otimizador_mobilenet(modelo, learning_rate: float):
    return torch.optim.AdamW(
        parametros_treinaveis(modelo),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )


def calcular_pesos_classes_mobilenet(df_treino: pd.DataFrame, dispositivo):
    contagens = df_treino["alvo"].value_counts().reindex([0, 1], fill_value=0).astype(float)
    if (contagens == 0).any():
        raise ValueError(f"Treino interno sem uma das classes: {contagens.to_dict()}")
    total = float(contagens.sum())
    pesos = total / (len(CLASSES) * contagens)
    return torch.tensor(pesos.to_numpy(), dtype=torch.float32, device=dispositivo)


def transferir_imagens(imagens, dispositivo):
    usar_channels_last = dispositivo.type == "cuda" and USAR_CHANNELS_LAST_CUDA
    if usar_channels_last:
        return imagens.to(
            dispositivo,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
    return imagens.to(dispositivo, non_blocking=(dispositivo.type == "cuda"))


def fase_para_epoca(epoca: int) -> tuple[str, float]:
    if epoca <= EPOCHS_BACKBONE_CONGELADO:
        return "backbone_congelado", LEARNING_RATE_CLASSIFICADOR
    return "ajuste_fino_ultimos_blocos", LEARNING_RATE_AJUSTE_FINO


def metricas_treino_mobilenet(alvos: list[int], predicoes: list[int], perdas: list[float]) -> dict:
    tn = int(((np.asarray(alvos) == 0) & (np.asarray(predicoes) == 0)).sum())
    fp = int(((np.asarray(alvos) == 0) & (np.asarray(predicoes) == 1)).sum())
    fn = int(((np.asarray(alvos) == 1) & (np.asarray(predicoes) == 0)).sum())
    tp = int(((np.asarray(alvos) == 1) & (np.asarray(predicoes) == 1)).sum())
    metricas = calcular_metricas_confusao(tn, fp, fn, tp)
    metricas["loss"] = float(sum(perdas) / max(len(perdas), 1))
    return metricas


def treinar_uma_epoca_mobilenet(modelo, carregador, criterio, otimizador, dispositivo, scaler):
    modelo.train()
    perdas = []
    alvos = []
    predicoes = []
    usar_amp = USAR_MIXED_PRECISION and dispositivo.type == "cuda"

    for imagens, y in carregador:
        imagens = transferir_imagens(imagens, dispositivo)
        y = y.to(dispositivo, non_blocking=(dispositivo.type == "cuda"))

        otimizador.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=dispositivo.type, enabled=usar_amp):
            saidas = modelo(imagens)
            perda = criterio(saidas, y)

        scaler.scale(perda).backward()
        scaler.step(otimizador)
        scaler.update()

        perdas.append(float(perda.item()))
        pred = torch.argmax(saidas.detach(), dim=1)
        alvos.extend(y.detach().cpu().tolist())
        predicoes.extend(pred.cpu().tolist())

    return metricas_treino_mobilenet(alvos, predicoes, perdas)


@torch.no_grad()
def avaliar_mobilenet(modelo, carregador, criterio, dispositivo):
    modelo.eval()
    perdas = []
    alvos = []
    predicoes = []
    usar_amp = USAR_MIXED_PRECISION and dispositivo.type == "cuda"

    for imagens, y in carregador:
        imagens = transferir_imagens(imagens, dispositivo)
        y = y.to(dispositivo, non_blocking=(dispositivo.type == "cuda"))

        with torch.amp.autocast(device_type=dispositivo.type, enabled=usar_amp):
            saidas = modelo(imagens)
            perda = criterio(saidas, y)

        perdas.append(float(perda.item()))
        pred = torch.argmax(saidas, dim=1)
        alvos.extend(y.cpu().tolist())
        predicoes.extend(pred.cpu().tolist())

    return metricas_treino_mobilenet(alvos, predicoes, perdas)


@torch.no_grad()
def obter_probabilidades_mobilenet(modelo, carregador, dispositivo) -> np.ndarray:
    modelo.eval()
    probabilidades = []
    usar_amp = USAR_MIXED_PRECISION and dispositivo.type == "cuda"

    for imagens, _ in carregador:
        imagens = transferir_imagens(imagens, dispositivo)
        with torch.amp.autocast(device_type=dispositivo.type, enabled=usar_amp):
            saidas = modelo(imagens)
            prob = torch.softmax(saidas, dim=1)[:, INDICE_POSITIVO]
        probabilidades.extend(prob.detach().cpu().tolist())

    return np.asarray(probabilidades, dtype=float)


def caminho_mobilenet_fold(fold: dict) -> tuple[Path, Path]:
    sufixo = f"fold_{int(fold['fold']):03d}_{nome_seguro(fold['grupo_externo'])}"
    caminho_checkpoint = PASTA_CHECKPOINTS / f"mobilenetv2_{sufixo}.pt"
    caminho_historico = PASTA_HISTORICOS_MOBILENET / f"historico_mobilenetv2_{sufixo}.csv"
    return caminho_checkpoint, caminho_historico


def treinar_mobilenet_fold(
    base: pd.DataFrame,
    fold: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configurar_semente_torch(SEMENTE_ALEATORIA)
    PASTA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    PASTA_HISTORICOS_MOBILENET.mkdir(parents=True, exist_ok=True)

    df_treino = base.loc[fold["indices_treino"]].copy()
    df_validacao = base.loc[fold["indices_validacao"]].copy()
    df_teste = base.loc[fold["indices_teste"]].copy()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformacao_treino, transformacao_validacao = criar_transformacoes_mobilenet()
    dataset_treino = DatasetRecortes(df_treino, transformacao_treino)
    dataset_validacao = DatasetRecortes(df_validacao, transformacao_validacao)
    dataset_teste = DatasetRecortes(df_teste, transformacao_validacao)

    carregador_treino = criar_data_loader_mobilenet(dataset_treino, shuffle=True)
    carregador_validacao = criar_data_loader_mobilenet(dataset_validacao, shuffle=False)
    carregador_teste = criar_data_loader_mobilenet(dataset_teste, shuffle=False)

    modelo = criar_modelo_mobilenetv2().to(dispositivo)
    if dispositivo.type == "cuda" and USAR_CHANNELS_LAST_CUDA:
        modelo = modelo.to(memory_format=torch.channels_last)

    pesos_classes = calcular_pesos_classes_mobilenet(df_treino, dispositivo)
    criterio = nn.CrossEntropyLoss(weight=pesos_classes)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(USAR_MIXED_PRECISION and dispositivo.type == "cuda"),
    )

    caminho_checkpoint, caminho_historico = caminho_mobilenet_fold(fold)
    melhor_loss = float("inf")
    melhor_epoca = 0
    melhor_fase = None
    epocas_sem_melhora = 0
    fase_atual = None
    otimizador = None
    historico = []
    inicio = time.time()

    for epoca in range(1, EPOCHS_TOTAL + 1):
        fase, learning_rate = fase_para_epoca(epoca)
        if fase != fase_atual:
            fase_anterior = fase_atual
            fase_atual = fase
            if fase == "backbone_congelado":
                congelar_backbone(modelo)
            else:
                liberar_ultimos_blocos(modelo)
                if fase_anterior == "backbone_congelado":
                    epocas_sem_melhora = 0
            otimizador = criar_otimizador_mobilenet(modelo, learning_rate)

        metricas_treino = treinar_uma_epoca_mobilenet(
            modelo,
            carregador_treino,
            criterio,
            otimizador,
            dispositivo,
            scaler,
        )
        metricas_validacao = avaliar_mobilenet(
            modelo,
            carregador_validacao,
            criterio,
            dispositivo,
        )

        historico.append({
            "fold": int(fold["fold"]),
            "grupo_externo": fold["grupo_externo"],
            "grupo_validacao": fold["grupo_validacao"],
            "epoca": epoca,
            "fase": fase,
            "learning_rate": learning_rate,
            **{f"treino_{k}": v for k, v in metricas_treino.items()},
            **{f"validacao_{k}": v for k, v in metricas_validacao.items()},
        })

        loss_validacao = float(metricas_validacao["loss"])
        if loss_validacao < melhor_loss:
            melhor_loss = loss_validacao
            melhor_epoca = epoca
            melhor_fase = fase
            epocas_sem_melhora = 0
            checkpoint = {
                "modelo": "mobilenet_v2",
                "nome_modelo": "mobilenetv2_recortes",
                "protocolo": PROTOCOLO,
                "fold": int(fold["fold"]),
                "grupo_externo": fold["grupo_externo"],
                "grupo_validacao": fold["grupo_validacao"],
                "pesos_pre_treinados": PESOS_PRE_TREINADOS,
                "pesos_imagenet_carregados": PESOS_IMAGENET_CARREGADOS,
                "state_dict": modelo.state_dict(),
                "classes": CLASSES,
                "classe_positiva": CLASSE_POSITIVA,
                "class_to_idx": CLASSE_PARA_INDICE,
                "tamanho_imagem": TAMANHO_IMAGEM,
                "seed": SEMENTE_ALEATORIA,
                "epoca": epoca,
                "fase": fase,
                "melhor_loss_validacao": melhor_loss,
                "metricas_validacao": metricas_validacao,
                "data_treino": datetime.now().isoformat(timespec="seconds"),
            }
            torch.save(checkpoint, caminho_checkpoint)
        else:
            epocas_sem_melhora += 1

        if epocas_sem_melhora >= PACIENCIA_EARLY_STOPPING:
            break

    pd.DataFrame(historico).to_csv(caminho_historico, index=False, encoding="utf-8-sig")
    checkpoint = torch.load(caminho_checkpoint, map_location=dispositivo)
    modelo.load_state_dict(checkpoint["state_dict"])
    tempo_treino = time.time() - inicio

    prob_validacao = obter_probabilidades_mobilenet(modelo, carregador_validacao, dispositivo)
    prob_teste = obter_probabilidades_mobilenet(modelo, carregador_teste, dispositivo)
    contexto = completar_contexto_fold(contexto_modelo("mobilenetv2"), fold)

    parametros_json = json.dumps(
        {
            "pesos_pre_treinados": PESOS_PRE_TREINADOS,
            "pesos_imagenet_carregados": PESOS_IMAGENET_CARREGADOS,
            "entrada": f"{TAMANHO_IMAGEM}x{TAMANHO_IMAGEM}",
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "mixed_precision": USAR_MIXED_PRECISION,
            "pin_memory": PIN_MEMORY,
            "persistent_workers": PERSISTENT_WORKERS,
            "epochs_total": EPOCHS_TOTAL,
            "epochs_backbone_congelado": EPOCHS_BACKBONE_CONGELADO,
            "paciencia_early_stopping": PACIENCIA_EARLY_STOPPING,
            "learning_rate_classificador": LEARNING_RATE_CLASSIFICADOR,
            "learning_rate_ajuste_fino": LEARNING_RATE_AJUSTE_FINO,
            "weight_decay": WEIGHT_DECAY,
            "blocos_finais_descongelados": BLOCOS_FINAIS_DESCONGELADOS,
            "checkpoint": str(caminho_checkpoint.relative_to(PASTA_PROJETO)),
            "historico": str(caminho_historico.relative_to(PASTA_PROJETO)),
            "melhor_fase": melhor_fase,
            "pesos_classes": pesos_classes.detach().cpu().tolist(),
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
        sort_keys=True,
    )

    return avaliar_probabilidades(
        df_validacao,
        df_teste,
        prob_validacao,
        prob_teste,
        contexto,
        tempo_treino,
        melhor_epoca=melhor_epoca,
        melhor_loss_validacao=melhor_loss,
        parametros_json=parametros_json,
    )


def avaliar_baseline_sempre_contaminada(
    base: pd.DataFrame,
    fold: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_teste = base.loc[fold["indices_teste"]].copy()
    contexto = completar_contexto_fold(contexto_modelo("baseline_sempre_contaminada"), fold)
    y_teste = df_teste["alvo"].to_numpy(dtype=int)
    pred = np.ones(len(df_teste), dtype=int)

    tn = int(((y_teste == 0) & (pred == 0)).sum())
    fp = int(((y_teste == 0) & (pred == 1)).sum())
    fn = int(((y_teste == 1) & (pred == 0)).sum())
    tp = int(((y_teste == 1) & (pred == 1)).sum())
    metricas = pd.DataFrame([{
        **contexto,
        "cenario": "teste_baseline_sempre_contaminada",
        "threshold": "nao_aplicavel",
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        **calcular_metricas_confusao(tn, fp, fn, tp),
        "tempo_treino_segundos": 0.0,
        "melhor_epoca": None,
        "melhor_loss_validacao": None,
        "parametros_json": json.dumps({"regra": "prediz_todas_contaminadas"}),
    }])

    predicoes = []
    for _, linha in df_teste.iterrows():
        predicoes.append({
            **contexto,
            "cenario": "teste_baseline_sempre_contaminada",
            "threshold": "nao_aplicavel",
            "nome_arquivo": linha["nome_arquivo"],
            "caminho_relativo": linha["caminho_relativo"],
            "classe_real": linha["classe"],
            "alvo": int(linha["alvo"]),
            "prob_contaminada": 1.0,
            "predicao": "contaminada",
            "papel_amostra": "teste_externo",
            "split_original": linha["split_original"],
        })
    thresholds = pd.DataFrame([{
        **contexto,
        "cenario": "teste_baseline_sempre_contaminada",
        "threshold": "nao_aplicavel",
    }])
    return metricas, pd.DataFrame(predicoes), thresholds


def carregar_csv_existente(caminho: Path) -> pd.DataFrame:
    if caminho.exists():
        return pd.read_csv(caminho)
    return pd.DataFrame()


def resultado_completo(metricas_existentes: pd.DataFrame, fold: dict, modelo: str) -> bool:
    if metricas_existentes.empty:
        return False
    contexto = contexto_modelo(modelo)
    modelo_saida = contexto["modelo"]
    cenarios_esperados = (
        {"teste_baseline_sempre_contaminada"}
        if modelo == "baseline_sempre_contaminada"
        else {
            "teste_threshold_0_50",
            "teste_threshold_melhor_f1_validacao",
            "teste_threshold_prioridade_recall_validacao",
        }
    )
    linhas = metricas_existentes[
        (metricas_existentes["fold"].astype(int) == int(fold["fold"]))
        & (metricas_existentes["modelo"].astype(str) == modelo_saida)
    ]
    return cenarios_esperados.issubset(set(linhas["cenario"].astype(str)))


def concatenar_salvar(
    caminho: Path,
    novos: list[pd.DataFrame],
    chaves: list[str],
    retomar: bool,
) -> pd.DataFrame:
    tabelas = []
    existente = carregar_csv_existente(caminho) if retomar else pd.DataFrame()
    if not existente.empty:
        tabelas.append(existente)
    tabelas.extend([df for df in novos if not df.empty])
    if tabelas:
        saida = pd.concat(tabelas, ignore_index=True, sort=False)
        chaves_presentes = [chave for chave in chaves if chave in saida.columns]
        if chaves_presentes:
            saida = saida.drop_duplicates(chaves_presentes, keep="last")
    else:
        saida = pd.DataFrame()
    saida.to_csv(caminho, index=False, encoding="utf-8-sig")
    return saida


def resumo_micro_macro(metricas: pd.DataFrame) -> pd.DataFrame:
    if metricas.empty:
        return pd.DataFrame()

    grupos = [
        "modelo",
        "familia_modelo",
        "tipo_entrada",
        "cenario",
        "conjunto_features",
        "resultado_oficial",
        "papel_experimento",
    ]
    registros = []

    for chaves, df_grupo in metricas.groupby(grupos, dropna=False):
        base = dict(zip(grupos, chaves))
        tn = int(df_grupo["tn"].sum())
        fp = int(df_grupo["fp"].sum())
        fn = int(df_grupo["fn"].sum())
        tp = int(df_grupo["tp"].sum())
        registros.append({
            **base,
            "agregacao": "micro",
            "folds": int(df_grupo["fold"].nunique()),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            **calcular_metricas_confusao(tn, fp, fn, tp),
        })

        linha_macro = {
            **base,
            "agregacao": "macro",
            "folds": int(df_grupo["fold"].nunique()),
            "tn": np.nan,
            "fp": np.nan,
            "fn": np.nan,
            "tp": np.nan,
        }
        for coluna in COLUNAS_METRICAS:
            linha_macro[f"{coluna}_media"] = float(df_grupo[coluna].mean())
            linha_macro[f"{coluna}_dp"] = float(df_grupo[coluna].std(ddof=1))
        linha_macro["total_media"] = float(df_grupo["total"].mean())
        linha_macro["total_dp"] = float(df_grupo["total"].std(ddof=1))
        registros.append(linha_macro)

    return pd.DataFrame(registros)


def gerar_comparacao_split_original(resumo: pd.DataFrame) -> pd.DataFrame:
    if not CAMINHO_COMPARACAO_SPLIT_ORIGINAL.exists() or resumo.empty:
        return pd.DataFrame()

    original = pd.read_csv(CAMINHO_COMPARACAO_SPLIT_ORIGINAL)
    externo = resumo[resumo["agregacao"].astype(str) == "micro"].copy()
    chaves = ["modelo", "conjunto_features", "cenario"]
    for chave in chaves:
        if chave not in original.columns:
            original[chave] = CONJUNTO_NAO_APLICAVEL if chave == "conjunto_features" else ""
        if chave not in externo.columns:
            externo[chave] = CONJUNTO_NAO_APLICAVEL if chave == "conjunto_features" else ""

    comparacao = externo.merge(
        original,
        on=chaves,
        how="inner",
        suffixes=("_tratamento", "_split_original"),
    )
    if comparacao.empty:
        return comparacao

    comparacao["observacao"] = (
        "Comparacao descritiva: o protocolo leave-one-group-out por tratamento "
        "difere do split aleatorio original."
    )
    for coluna in COLUNAS_COMPARACAO_PROTOCOLOS:
        comparacao[f"delta_{coluna}"] = (
            comparacao[f"{coluna}_tratamento"] - comparacao[f"{coluna}_split_original"]
        )
    return comparacao


def validar_auditoria_predicoes(predicoes: pd.DataFrame, folds_df: pd.DataFrame):
    if predicoes.empty:
        return
    esperados_por_fold = (
        folds_df[folds_df["papel_amostra"] == "teste_externo"]
        .groupby("fold")["nome_arquivo"]
        .apply(set)
        .to_dict()
    )
    for (modelo, cenario, fold), grupo in predicoes.groupby(["modelo", "cenario", "fold"]):
        esperados = esperados_por_fold.get(fold, set())
        contagens = grupo["nome_arquivo"].value_counts()
        duplicados = contagens[contagens != 1]
        faltantes = sorted(esperados - set(grupo["nome_arquivo"]))
        if not duplicados.empty or faltantes:
            raise ValueError(
                "Auditoria de predicoes falhou para "
                f"modelo={modelo}, cenario={cenario}, fold={fold}. "
                f"duplicados={duplicados.head(20).to_dict()} faltantes={faltantes[:20]}"
            )


def salvar_config(args, modelos: list[str], features: list[str], folds: list[dict], base: pd.DataFrame):
    config = {
        "data_execucao": datetime.now().isoformat(timespec="seconds"),
        "protocolo": PROTOCOLO,
        "grupo_principal": GRUPO_VALIDACAO_EXTERNA,
        "grupo_principal_definicao": "normalizar(experimento_rotulo) + '__' + normalizar(tratamento_planilha)",
        "preflight": bool(args.preflight),
        "modelos_solicitados": modelos,
        "somente_grupo": args.somente_grupo,
        "retomar": bool(args.retomar),
        "seed": SEMENTE_ALEATORIA,
        "total_amostras": int(len(base)),
        "total_grupos": int(base[GRUPO_VALIDACAO_EXTERNA].nunique()),
        "folds_executaveis": int(len(folds)),
        "features_classicos": {
            "conjunto": CONJUNTO_PRINCIPAL,
            "quantidade": len(features),
            "arquivo": str(CAMINHO_FEATURES.relative_to(PASTA_PROJETO)),
        },
        "random_forest": {
            "grid_igual_script_23": True,
            "cv": "StratifiedGroupKFold",
            "cv_folds_max": CV_FOLDS_MAX,
            "n_jobs_grid": N_JOBS_GRID,
            "n_jobs_random_forest": 1,
        },
        "svm_rbf": {
            "grid_igual_script_23": True,
            "pipeline": "SimpleImputer + StandardScaler + SVC RBF",
            "class_weight": "balanced",
        },
        "metadados": {
            "logica_base": "script_26_taxas_suavizadas",
            "alpha_suavizacao": ALPHA_SUAVIZACAO,
            "papel_experimento": "diagnostico_vies",
        },
        "mobilenetv2": {
            "pesos_pre_treinados": PESOS_PRE_TREINADOS,
            "pesos_imagenet_carregados": PESOS_IMAGENET_CARREGADOS,
            "entrada": f"{TAMANHO_IMAGEM}x{TAMANHO_IMAGEM}",
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "mixed_precision": USAR_MIXED_PRECISION,
            "pin_memory": PIN_MEMORY,
            "persistent_workers": PERSISTENT_WORKERS,
            "epochs_total": EPOCHS_TOTAL,
            "epochs_backbone_congelado": EPOCHS_BACKBONE_CONGELADO,
            "paciencia_early_stopping": PACIENCIA_EARLY_STOPPING,
            "learning_rate_classificador": LEARNING_RATE_CLASSIFICADOR,
            "learning_rate_ajuste_fino": LEARNING_RATE_AJUSTE_FINO,
            "weight_decay": WEIGHT_DECAY,
            "blocos_finais_descongelados": BLOCOS_FINAIS_DESCONGELADOS,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
        "thresholds": {
            "fixo": 0.50,
            "melhor_f1": "selecionado somente na validacao interna",
            "prioridade_recall": f"recall >= {RECALL_MINIMO_PRIORITARIO}, maior F1, maior especificidade",
        },
        "arquivos_saida": {
            "folds": str(CAMINHO_FOLDS.relative_to(PASTA_PROJETO)),
            "predicoes": str(CAMINHO_PREDICOES.relative_to(PASTA_PROJETO)),
            "metricas": str(CAMINHO_METRICAS.relative_to(PASTA_PROJETO)),
            "thresholds": str(CAMINHO_THRESHOLDS.relative_to(PASTA_PROJETO)),
            "resumo": str(CAMINHO_RESUMO.relative_to(PASTA_PROJETO)),
            "comparacao_split_original": str(CAMINHO_COMPARACAO_PROTOCOLOS.relative_to(PASTA_PROJETO)),
            "diagnostico_folds": str(CAMINHO_DIAGNOSTICO_FOLDS.relative_to(PASTA_PROJETO)),
        },
    }
    CAMINHO_CONFIG.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def executar_modelo(
    nome_modelo: str,
    base: pd.DataFrame,
    features: list[str],
    fold: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if nome_modelo in {"random_forest", "svm_rbf"}:
        return treinar_classico_fold(base, features, fold, nome_modelo)
    if nome_modelo == "metadados":
        return treinar_metadados_fold(base, fold)
    if nome_modelo == "mobilenetv2":
        return treinar_mobilenet_fold(base, fold)
    if nome_modelo == "baseline_sempre_contaminada":
        return avaliar_baseline_sempre_contaminada(base, fold)
    raise ValueError(f"Modelo desconhecido: {nome_modelo}")


def main():
    parser = criar_parser()
    args = parser.parse_args()
    modelos = modelos_solicitados(args.modelos)

    print("=" * 70)
    print("VALIDACAO EXTERNA POR EXPERIMENTO_TRATAMENTO")
    print("=" * 70)

    PASTA_VALIDACAO.mkdir(parents=True, exist_ok=True)
    base = carregar_base_experimento()
    features = carregar_features_principais(base)
    folds, folds_df, diagnosticos_df = criar_folds(base, args.somente_grupo)

    folds_df.to_csv(CAMINHO_FOLDS, index=False, encoding="utf-8-sig")
    diagnosticos_df.to_csv(CAMINHO_DIAGNOSTICO_FOLDS, index=False, encoding="utf-8-sig")
    salvar_config(args, modelos, features, folds, base)
    validar_folds_antes_do_treino(diagnosticos_df)

    print(f"Amostras: {len(base)}")
    print(f"Grupos: {base[GRUPO_VALIDACAO_EXTERNA].nunique()}")
    print(f"Folds nesta execucao: {len(folds)}")
    print(f"Features visuais principais: {len(features)}")

    if args.preflight:
        print("Preflight concluido. Nenhum modelo foi treinado.")
        print(f"Folds: {CAMINHO_FOLDS}")
        print(f"Diagnostico: {CAMINHO_DIAGNOSTICO_FOLDS}")
        print(f"Config: {CAMINHO_CONFIG}")
        return

    metricas_existentes = carregar_csv_existente(CAMINHO_METRICAS) if args.retomar else pd.DataFrame()
    metricas_novas = []
    predicoes_novas = []
    thresholds_novos = []
    modelos_execucao = [*modelos, "baseline_sempre_contaminada"]

    for fold in folds:
        print()
        print(
            f"Fold {fold['fold']} | externo={fold['grupo_externo']} | "
            f"validacao={fold['grupo_validacao']}"
        )
        for nome_modelo in modelos_execucao:
            if args.retomar and resultado_completo(metricas_existentes, fold, nome_modelo):
                print(f"- {nome_modelo}: ja completo, pulando.")
                continue
            print(f"- {nome_modelo}: executando")
            metricas, predicoes, thresholds = executar_modelo(nome_modelo, base, features, fold)
            metricas_novas.append(metricas)
            predicoes_novas.append(predicoes)
            thresholds_novos.append(thresholds)

    metricas = concatenar_salvar(
        CAMINHO_METRICAS,
        metricas_novas,
        ["modelo", "fold", "cenario", "conjunto_features"],
        args.retomar,
    )
    predicoes = concatenar_salvar(
        CAMINHO_PREDICOES,
        predicoes_novas,
        ["modelo", "fold", "cenario", "nome_arquivo", "conjunto_features"],
        args.retomar,
    )
    thresholds = concatenar_salvar(
        CAMINHO_THRESHOLDS,
        thresholds_novos,
        ["modelo", "fold", "threshold", "conjunto_features"],
        args.retomar,
    )

    validar_auditoria_predicoes(predicoes, folds_df)
    resumo = resumo_micro_macro(metricas)
    resumo.to_csv(CAMINHO_RESUMO, index=False, encoding="utf-8-sig")

    comparacao = gerar_comparacao_split_original(resumo)
    comparacao.to_csv(CAMINHO_COMPARACAO_PROTOCOLOS, index=False, encoding="utf-8-sig")

    print()
    print("Arquivos gerados:")
    for caminho in [
        CAMINHO_FOLDS,
        CAMINHO_PREDICOES,
        CAMINHO_METRICAS,
        CAMINHO_THRESHOLDS,
        CAMINHO_RESUMO,
        CAMINHO_COMPARACAO_PROTOCOLOS,
        CAMINHO_CONFIG,
        CAMINHO_DIAGNOSTICO_FOLDS,
    ]:
        print(f"- {caminho}")


if __name__ == "__main__":
    main()
