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

from .thresholds import avaliar_probabilidades



def logit(probabilidade) -> np.ndarray:
    p = np.clip(probabilidade, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(valor) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(valor)))


def ajustar_transformador_metadados(treino: pd.DataFrame) -> dict:
    categorias = {}
    for coluna in COLUNAS_CATEGORICAS_METADADOS:
        categorias[coluna] = sorted(
            treino[coluna].fillna("desconhecido").astype(str).unique().tolist()
        )

    numericas = {}
    for coluna in COLUNAS_NUMERICAS_METADADOS:
        valores = pd.to_numeric(treino[coluna], errors="coerce").dropna()
        if valores.empty:
            numericas[coluna] = {"tipo": "sem_valor"}
            continue

        valores_unicos = sorted(float(valor) for valor in valores.unique().tolist())
        if len(valores_unicos) <= 8:
            numericas[coluna] = {"tipo": "valores", "valores": valores_unicos}
            continue

        try:
            _, bins = pd.qcut(valores, q=5, duplicates="drop", retbins=True)
            bins = np.unique(bins.astype(float))
            if len(bins) < 2:
                numericas[coluna] = {"tipo": "valores", "valores": valores_unicos}
            else:
                bins[0] = -np.inf
                bins[-1] = np.inf
                numericas[coluna] = {"tipo": "bins", "bins": bins.tolist()}
        except ValueError:
            numericas[coluna] = {"tipo": "valores", "valores": valores_unicos}

    return {"categorias": categorias, "numericas": numericas}


def aplicar_transformador_metadados(df: pd.DataFrame, transformador: dict) -> pd.DataFrame:
    saida = df.copy()
    for coluna, categorias in transformador["categorias"].items():
        permitidas = set(categorias)
        valores = saida[coluna].fillna("desconhecido").astype(str)
        saida[coluna] = valores.where(valores.isin(permitidas), "categoria_nao_vista_treino")

    for coluna, regra in transformador["numericas"].items():
        valores = pd.to_numeric(saida[coluna], errors="coerce")
        coluna_faixa = f"{coluna}_faixa"
        if regra["tipo"] == "sem_valor":
            saida[coluna_faixa] = "sem_valor"
        elif regra["tipo"] == "valores":
            permitidos = set(float(valor) for valor in regra["valores"])
            saida[coluna_faixa] = valores.map(
                lambda valor: "sem_valor"
                if pd.isna(valor)
                else (
                    f"valor_{valor:g}"
                    if float(valor) in permitidos
                    else "valor_fora_treino"
                )
            )
        else:
            bins = np.asarray(regra["bins"], dtype=float)
            labels = [f"bin_{indice:02d}" for indice in range(len(bins) - 1)]
            saida[coluna_faixa] = pd.cut(
                valores,
                bins=bins,
                labels=labels,
                include_lowest=True,
            ).astype(str)
            saida[coluna_faixa] = saida[coluna_faixa].replace("nan", "sem_valor")

    return saida


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
    df_treino_bruto = base.loc[fold["indices_treino"]].copy()
    df_validacao_bruto = base.loc[fold["indices_validacao"]].copy()
    df_teste_bruto = base.loc[fold["indices_teste"]].copy()

    transformador = ajustar_transformador_metadados(df_treino_bruto)
    df_treino = aplicar_transformador_metadados(df_treino_bruto, transformador)
    df_validacao = aplicar_transformador_metadados(df_validacao_bruto, transformador)
    df_teste = aplicar_transformador_metadados(df_teste_bruto, transformador)

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
                "transformador_fit_somente_treino_interno": True,
                "categorias_por_campo": {
                    chave: len(valor)
                    for chave, valor in transformador["categorias"].items()
                },
                "numeric_rules": transformador["numericas"],
            },
            sort_keys=True,
        ),
    )
