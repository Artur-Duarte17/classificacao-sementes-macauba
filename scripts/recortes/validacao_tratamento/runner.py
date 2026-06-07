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

from .agregacao import gerar_comparacao_split_original, resumo_micro_macro, validar_auditoria_predicoes
from .classicos import treinar_classico_fold
from .controles import avaliar_baseline_sempre_contaminada
from .dados import carregar_base_experimento, carregar_features_principais
from .folds import criar_folds, validar_folds_antes_do_treino
from .metadados import treinar_metadados_fold
from .mobilenet import treinar_mobilenet_fold
from .persistencia import (
    atualizar_csv_incremental,
    carregar_csv_existente,
    gravar_csv_atomico,
    resultado_completo,
    salvar_config,
)



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


def atualizar_agregados(metricas: pd.DataFrame):
    resumo = resumo_micro_macro(metricas)
    gravar_csv_atomico(resumo, CAMINHO_RESUMO)
    comparacao = gerar_comparacao_split_original(resumo)
    gravar_csv_atomico(comparacao, CAMINHO_COMPARACAO_PROTOCOLOS)


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

    gravar_csv_atomico(folds_df, CAMINHO_FOLDS)
    gravar_csv_atomico(diagnosticos_df, CAMINHO_DIAGNOSTICO_FOLDS)
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
    predicoes_existentes = carregar_csv_existente(CAMINHO_PREDICOES) if args.retomar else pd.DataFrame()
    thresholds_existentes = carregar_csv_existente(CAMINHO_THRESHOLDS) if args.retomar else pd.DataFrame()
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
            metricas_existentes = atualizar_csv_incremental(
                CAMINHO_METRICAS,
                metricas,
                CHAVES_RESULTADO,
                metricas_existentes,
            )
            predicoes_existentes = atualizar_csv_incremental(
                CAMINHO_PREDICOES,
                predicoes,
                CHAVES_PREDICAO,
                predicoes_existentes,
            )
            thresholds_existentes = atualizar_csv_incremental(
                CAMINHO_THRESHOLDS,
                thresholds,
                CHAVES_THRESHOLD,
                thresholds_existentes,
            )
            validar_auditoria_predicoes(predicoes_existentes, folds_df)
            atualizar_agregados(metricas_existentes)

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
        CAMINHO_DIAGNOSTICO_CV,
    ]:
        print(f"- {caminho}")
