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

from .metricas import calcular_metricas_confusao



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
            "folds": int(df_grupo["grupo_externo"].nunique()),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            **calcular_metricas_confusao(tn, fp, fn, tp),
        })

        linha_macro = {
            **base,
            "agregacao": "macro",
            "folds": int(df_grupo["grupo_externo"].nunique()),
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
    folds_teste = folds_df[folds_df["papel_amostra"] == "teste_externo"].copy()
    grupos_escopo = set(folds_teste["grupo_externo"].astype(str))
    predicoes_escopo = predicoes[
        predicoes["grupo_externo"].astype(str).isin(grupos_escopo)
    ].copy()
    esperados_por_grupo = (
        folds_teste
        .groupby("grupo_externo")["nome_arquivo"]
        .apply(set)
        .to_dict()
    )
    for (modelo, cenario, conjunto_features, grupo_externo), grupo in predicoes_escopo.groupby(
        ["modelo", "cenario", "conjunto_features", "grupo_externo"]
    ):
        esperados = esperados_por_grupo.get(grupo_externo, set())
        contagens = grupo["nome_arquivo"].value_counts()
        duplicados = contagens[contagens != 1]
        faltantes = sorted(esperados - set(grupo["nome_arquivo"]))
        extras = sorted(set(grupo["nome_arquivo"]) - esperados)
        if not duplicados.empty or faltantes or extras:
            raise ValueError(
                "Auditoria de predicoes falhou para "
                f"modelo={modelo}, cenario={cenario}, features={conjunto_features}, "
                f"grupo_externo={grupo_externo}. "
                f"duplicados={duplicados.head(20).to_dict()} "
                f"faltantes={faltantes[:20]} extras={extras[:20]}"
            )
