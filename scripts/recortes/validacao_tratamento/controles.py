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
