from __future__ import annotations

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

from .config import *



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
