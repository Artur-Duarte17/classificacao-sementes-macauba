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
