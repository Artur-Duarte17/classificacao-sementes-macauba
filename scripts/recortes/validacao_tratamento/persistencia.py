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



def gravar_csv_atomico(df: pd.DataFrame, caminho: Path):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(caminho.name + ".tmp")
    if df.empty and len(df.columns) == 0:
        temporario.write_text("", encoding="utf-8")
    else:
        df.to_csv(temporario, index=False, encoding="utf-8-sig")
        pd.read_csv(temporario)
    temporario.replace(caminho)


def gravar_json_atomico(objeto: dict, caminho: Path):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(caminho.name + ".tmp")
    temporario.write_text(
        json.dumps(objeto, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    json.loads(temporario.read_text(encoding="utf-8"))
    temporario.replace(caminho)


def carregar_csv_existente(caminho: Path) -> pd.DataFrame:
    if caminho.exists():
        try:
            return pd.read_csv(caminho)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
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
        (metricas_existentes["modelo"].astype(str) == modelo_saida)
        & (metricas_existentes["grupo_externo"].astype(str) == str(fold["grupo_externo"]))
        & (
            metricas_existentes["conjunto_features"].astype(str)
            == str(contexto["conjunto_features"])
        )
    ]
    return cenarios_esperados.issubset(set(linhas["cenario"].astype(str)))


def atualizar_csv_incremental(
    caminho: Path,
    novo: pd.DataFrame,
    chaves: list[str],
    existente: pd.DataFrame | None = None,
) -> pd.DataFrame:
    tabelas = []
    if existente is None:
        existente = carregar_csv_existente(caminho)
    if not existente.empty:
        tabelas.append(existente)
    if novo is not None and not novo.empty:
        tabelas.append(novo)
    if tabelas:
        saida = pd.concat(tabelas, ignore_index=True, sort=False)
        chaves_presentes = [chave for chave in chaves if chave in saida.columns]
        if chaves_presentes:
            saida = saida.drop_duplicates(chaves_presentes, keep="last")
    else:
        saida = pd.DataFrame()
    gravar_csv_atomico(saida, caminho)
    return saida


def registrar_diagnostico_cv(fold: dict, modelo: str, diagnosticos: list[dict]):
    novo = pd.DataFrame([
        {
            "modelo": modelo,
            "fold": int(fold["fold"]),
            "grupo_externo": fold["grupo_externo"],
            "grupo_validacao": fold["grupo_validacao"],
            **item,
        }
        for item in diagnosticos
    ])
    existente = carregar_csv_existente(CAMINHO_DIAGNOSTICO_CV)
    atualizar_csv_incremental(
        CAMINHO_DIAGNOSTICO_CV,
        novo,
        ["modelo", "grupo_externo", "n_splits", "divisao"],
        existente,
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
            "cv_folds_tentativa": [5, 4, 3, 2],
            "n_jobs_grid": N_JOBS_GRID,
            "n_jobs_random_forest": 1,
        },
        "svm_rbf": {
            "grid_igual_script_23": True,
            "pipeline": "SimpleImputer + StandardScaler + SVC RBF",
            "class_weight": "balanced",
        },
        "knn": {
            "grid_igual_script_23": True,
            "pipeline": "SimpleImputer + StandardScaler + KNeighborsClassifier",
            "n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
            "n_neighbors_filtrado_por_menor_treino_cv": True,
            "weights": ["uniform", "distance"],
            "p": [1, 2],
            "algorithm": "auto",
        },
        "lda": {
            "grid_igual_script_23": True,
            "pipeline": "SimpleImputer + StandardScaler + LinearDiscriminantAnalysis",
            "svd": {"tol": [1e-4, 1e-3, 1e-2]},
            "lsqr": {"shrinkage": [None, "auto", 0.01, 0.1, 0.5, 0.9]},
            "eigen": False,
        },
        "metadados": {
            "logica_base": "script_26_taxas_suavizadas",
            "alpha_suavizacao": ALPHA_SUAVIZACAO,
            "papel_experimento": "diagnostico_vies",
            "fit_apply_por_fold": True,
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
            "diagnostico_cv_interna": str(CAMINHO_DIAGNOSTICO_CV.relative_to(PASTA_PROJETO)),
        },
    }
    gravar_json_atomico(config, CAMINHO_CONFIG)
