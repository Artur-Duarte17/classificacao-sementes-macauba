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

from .dados import normalizar_texto
from .persistencia import gravar_csv_atomico



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
    grupos_globais = sorted(base[GRUPO_VALIDACAO_EXTERNA].astype(str).unique().tolist())
    mapa_folds = {grupo: indice + 1 for indice, grupo in enumerate(grupos_globais)}
    grupos = grupos_globais
    if somente_grupo:
        grupo_normalizado = normalizar_texto(somente_grupo)
        if somente_grupo in grupos_globais:
            grupos = [somente_grupo]
        elif grupo_normalizado in grupos_globais:
            grupos = [grupo_normalizado]
        else:
            raise ValueError(
                f"Grupo solicitado nao encontrado: {somente_grupo}. "
                f"Exemplos disponiveis: {grupos_globais[:20]}"
            )

    folds = []
    diagnosticos = []
    registros_folds = []

    for grupo_externo in grupos:
        fold_id = mapa_folds[grupo_externo]
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
        gravar_csv_atomico(diagnosticos_df, CAMINHO_DIAGNOSTICO_FOLDS)
        raise ValueError(
            "Folds invalidos detectados antes do treino. "
            f"Diagnostico salvo em: {CAMINHO_DIAGNOSTICO_FOLDS}"
        )
