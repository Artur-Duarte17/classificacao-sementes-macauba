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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .config import *

from .dados import preparar_matriz
from .persistencia import registrar_diagnostico_cv
from .thresholds import avaliar_probabilidades



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


def criar_estimador_classico(nome_modelo: str, menor_treino_cv: int | None = None):
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

    if nome_modelo == "knn":
        candidatos_k = [3, 5, 7, 9, 11, 15, 21, 31]
        if menor_treino_cv is not None:
            candidatos_k = [valor for valor in candidatos_k if valor <= menor_treino_cv]
        if not candidatos_k:
            raise ValueError(
                "Nenhum valor de n_neighbors e compativel com a CV interna "
                f"para {nome_modelo}."
            )
        estimador = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("modelo", KNeighborsClassifier(algorithm="auto")),
        ])
        grid = {
            "modelo__n_neighbors": candidatos_k,
            "modelo__weights": ["uniform", "distance"],
            "modelo__p": [1, 2],
        }
        return estimador, grid

    if nome_modelo == "lda":
        estimador = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("modelo", LinearDiscriminantAnalysis()),
        ])
        grid = [
            {
                "modelo__solver": ["svd"],
                "modelo__tol": [1e-4, 1e-3, 1e-2],
            },
            {
                "modelo__solver": ["lsqr"],
                "modelo__shrinkage": [None, "auto", 0.01, 0.1, 0.5, 0.9],
            },
        ]
        return estimador, grid

    raise ValueError(f"Modelo classico desconhecido: {nome_modelo}")


def selecionar_cv_valido(df_treino: pd.DataFrame, fold: dict, nome_modelo: str):
    y = df_treino["alvo"].to_numpy(dtype=int)
    grupos = df_treino[GRUPO_VALIDACAO_EXTERNA].astype(str).to_numpy()
    x_dummy = np.zeros((len(df_treino), 1))
    diagnosticos = []

    for n_splits in [5, 4, 3, 2]:
        item_base = {"n_splits": n_splits, "valido": False, "motivo": ""}
        if len(set(grupos)) < n_splits:
            diagnosticos.append({
                **item_base,
                "divisao": "precheck",
                "motivo": "grupos_insuficientes",
            })
            continue
        if pd.Series(y).value_counts().reindex([0, 1], fill_value=0).min() < n_splits:
            diagnosticos.append({
                **item_base,
                "divisao": "precheck",
                "motivo": "classe_insuficiente",
            })
            continue

        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=SEMENTE_ALEATORIA,
        )
        valido = True
        try:
            splits = list(cv.split(x_dummy, y, groups=grupos))
        except ValueError as erro:
            diagnosticos.append({
                **item_base,
                "divisao": "split",
                "motivo": str(erro),
            })
            continue

        for indice_divisao, (idx_treino, idx_validacao) in enumerate(splits, start=1):
            y_treino = y[idx_treino]
            y_validacao = y[idx_validacao]
            grupos_treino = set(grupos[idx_treino])
            grupos_validacao = set(grupos[idx_validacao])
            motivos = []
            if pd.Series(y_treino).value_counts().reindex([0, 1], fill_value=0).min() == 0:
                motivos.append("treino_sem_duas_classes")
            if pd.Series(y_validacao).value_counts().reindex([0, 1], fill_value=0).min() == 0:
                motivos.append("validacao_sem_duas_classes")
            if grupos_treino & grupos_validacao:
                motivos.append("grupo_compartilhado")
            if motivos:
                valido = False
                diagnosticos.append({
                    **item_base,
                    "divisao": indice_divisao,
                    "motivo": ";".join(motivos),
                })
        if valido:
            diagnosticos.append({
                "n_splits": n_splits,
                "valido": True,
                "divisao": "todas",
                "motivo": "ok",
            })
            return cv, n_splits, diagnosticos

    registrar_diagnostico_cv(fold, nome_modelo, diagnosticos)
    raise ValueError(
        f"Nenhuma divisao StratifiedGroupKFold valida para {nome_modelo} "
        f"no grupo externo {fold['grupo_externo']}. Diagnostico: {CAMINHO_DIAGNOSTICO_CV}"
    )


def menor_tamanho_treino_cv(cv, x_treino: pd.DataFrame, y_treino: np.ndarray, grupos) -> int:
    tamanhos = [
        len(indices_treino)
        for indices_treino, _ in cv.split(x_treino, y_treino, groups=grupos)
    ]
    return int(min(tamanhos))


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

    cv, n_folds, diagnostico_cv = selecionar_cv_valido(df_treino, fold, nome_modelo)
    menor_treino_cv = menor_tamanho_treino_cv(cv, x_treino, y_treino, grupos_treino)
    estimador, grid = criar_estimador_classico(nome_modelo, menor_treino_cv)

    inicio = time.time()
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
            "diagnostico_cv": diagnostico_cv,
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
