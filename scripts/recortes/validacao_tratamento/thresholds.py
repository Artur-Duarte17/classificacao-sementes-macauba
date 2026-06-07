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

from .metricas import calcular_metricas_probabilidade



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
