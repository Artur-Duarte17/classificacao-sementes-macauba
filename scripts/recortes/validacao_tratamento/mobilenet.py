from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import gc
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

from .dados import nome_seguro
from .metricas import calcular_metricas_confusao
from .thresholds import avaliar_probabilidades



class DatasetRecortes(Dataset):
    def __init__(self, df: pd.DataFrame, transformacao):
        self.df = df.reset_index(drop=True)
        self.transformacao = transformacao

    def __len__(self):
        return len(self.df)

    def __getitem__(self, indice):
        linha = self.df.iloc[indice]
        caminho = PASTA_PROJETO / str(linha["caminho_relativo"])

        with Image.open(caminho) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

        imagem = self.transformacao(img)
        alvo = int(linha["alvo"])
        return imagem, alvo


def configurar_semente_torch(seed: int = SEMENTE_ALEATORIA):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def criar_transformacoes_mobilenet():
    media = [0.485, 0.456, 0.406]
    desvio = [0.229, 0.224, 0.225]

    transformacao_treino = transforms.Compose([
        transforms.RandomResizedCrop(TAMANHO_IMAGEM, scale=(0.80, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.15,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=media, std=desvio),
    ])

    transformacao_validacao = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(TAMANHO_IMAGEM),
        transforms.ToTensor(),
        transforms.Normalize(mean=media, std=desvio),
    ])

    return transformacao_treino, transformacao_validacao


def inicializar_worker_mobilenet(worker_id: int):
    seed = SEMENTE_ALEATORIA + worker_id
    random.seed(seed)
    np.random.seed(seed)


def criar_data_loader_mobilenet(dataset, shuffle: bool) -> DataLoader:
    usar_workers = NUM_WORKERS > 0
    gerador = torch.Generator()
    gerador.manual_seed(SEMENTE_ALEATORIA)
    opcoes = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS and usar_workers,
        "worker_init_fn": inicializar_worker_mobilenet if usar_workers else None,
        "generator": gerador,
    }
    if usar_workers:
        opcoes["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(dataset, **opcoes)


def criar_modelo_mobilenetv2():
    try:
        pesos = models.MobileNet_V2_Weights.DEFAULT
        modelo = models.mobilenet_v2(weights=pesos)
    except Exception as erro:
        raise RuntimeError(
            "Nao foi possivel carregar MobileNet_V2_Weights.DEFAULT. "
            "A validacao externa nao permite MobileNetV2 com weights=None."
        ) from erro

    entrada = modelo.classifier[1].in_features
    modelo.classifier[1] = nn.Linear(entrada, len(CLASSES))
    return modelo


def congelar_backbone(modelo):
    for parametro in modelo.features.parameters():
        parametro.requires_grad = False
    for parametro in modelo.classifier.parameters():
        parametro.requires_grad = True


def liberar_ultimos_blocos(modelo):
    for parametro in modelo.features.parameters():
        parametro.requires_grad = False
    blocos = list(modelo.features.children())[-BLOCOS_FINAIS_DESCONGELADOS:]
    for bloco in blocos:
        for parametro in bloco.parameters():
            parametro.requires_grad = True
    for parametro in modelo.classifier.parameters():
        parametro.requires_grad = True


def parametros_treinaveis(modelo):
    return [parametro for parametro in modelo.parameters() if parametro.requires_grad]


def criar_otimizador_mobilenet(modelo, learning_rate: float):
    return torch.optim.AdamW(
        parametros_treinaveis(modelo),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )


def calcular_pesos_classes_mobilenet(df_treino: pd.DataFrame, dispositivo):
    contagens = df_treino["alvo"].value_counts().reindex([0, 1], fill_value=0).astype(float)
    if (contagens == 0).any():
        raise ValueError(f"Treino interno sem uma das classes: {contagens.to_dict()}")
    total = float(contagens.sum())
    pesos = total / (len(CLASSES) * contagens)
    return torch.tensor(pesos.to_numpy(), dtype=torch.float32, device=dispositivo)


def transferir_imagens(imagens, dispositivo):
    usar_channels_last = dispositivo.type == "cuda" and USAR_CHANNELS_LAST_CUDA
    if usar_channels_last:
        return imagens.to(
            dispositivo,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
    return imagens.to(dispositivo, non_blocking=(dispositivo.type == "cuda"))


def fase_para_epoca(epoca: int) -> tuple[str, float]:
    if epoca <= EPOCHS_BACKBONE_CONGELADO:
        return "backbone_congelado", LEARNING_RATE_CLASSIFICADOR
    return "ajuste_fino_ultimos_blocos", LEARNING_RATE_AJUSTE_FINO


def metricas_treino_mobilenet(alvos: list[int], predicoes: list[int], perdas: list[float]) -> dict:
    tn = int(((np.asarray(alvos) == 0) & (np.asarray(predicoes) == 0)).sum())
    fp = int(((np.asarray(alvos) == 0) & (np.asarray(predicoes) == 1)).sum())
    fn = int(((np.asarray(alvos) == 1) & (np.asarray(predicoes) == 0)).sum())
    tp = int(((np.asarray(alvos) == 1) & (np.asarray(predicoes) == 1)).sum())
    metricas = calcular_metricas_confusao(tn, fp, fn, tp)
    metricas["loss"] = float(sum(perdas) / max(len(perdas), 1))
    return metricas


def treinar_uma_epoca_mobilenet(modelo, carregador, criterio, otimizador, dispositivo, scaler):
    modelo.train()
    perdas = []
    alvos = []
    predicoes = []
    usar_amp = USAR_MIXED_PRECISION and dispositivo.type == "cuda"

    for imagens, y in carregador:
        imagens = transferir_imagens(imagens, dispositivo)
        y = y.to(dispositivo, non_blocking=(dispositivo.type == "cuda"))

        otimizador.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=dispositivo.type, enabled=usar_amp):
            saidas = modelo(imagens)
            perda = criterio(saidas, y)

        scaler.scale(perda).backward()
        scaler.step(otimizador)
        scaler.update()

        perdas.append(float(perda.item()))
        pred = torch.argmax(saidas.detach(), dim=1)
        alvos.extend(y.detach().cpu().tolist())
        predicoes.extend(pred.cpu().tolist())

    return metricas_treino_mobilenet(alvos, predicoes, perdas)


def avaliar_mobilenet(modelo, carregador, criterio, dispositivo):
    modelo.eval()
    perdas = []
    alvos = []
    predicoes = []
    usar_amp = USAR_MIXED_PRECISION and dispositivo.type == "cuda"

    for imagens, y in carregador:
        imagens = transferir_imagens(imagens, dispositivo)
        y = y.to(dispositivo, non_blocking=(dispositivo.type == "cuda"))

        with torch.amp.autocast(device_type=dispositivo.type, enabled=usar_amp):
            saidas = modelo(imagens)
            perda = criterio(saidas, y)

        perdas.append(float(perda.item()))
        pred = torch.argmax(saidas, dim=1)
        alvos.extend(y.cpu().tolist())
        predicoes.extend(pred.cpu().tolist())

    return metricas_treino_mobilenet(alvos, predicoes, perdas)


def obter_probabilidades_mobilenet(modelo, carregador, dispositivo) -> np.ndarray:
    modelo.eval()
    probabilidades = []
    usar_amp = USAR_MIXED_PRECISION and dispositivo.type == "cuda"

    for imagens, _ in carregador:
        imagens = transferir_imagens(imagens, dispositivo)
        with torch.amp.autocast(device_type=dispositivo.type, enabled=usar_amp):
            saidas = modelo(imagens)
            prob = torch.softmax(saidas, dim=1)[:, INDICE_POSITIVO]
        probabilidades.extend(prob.detach().cpu().tolist())

    return np.asarray(probabilidades, dtype=float)


def caminho_mobilenet_fold(fold: dict) -> tuple[Path, Path]:
    sufixo = f"fold_{int(fold['fold']):03d}_{nome_seguro(fold['grupo_externo'])}"
    caminho_checkpoint = PASTA_CHECKPOINTS / f"mobilenetv2_{sufixo}.pt"
    caminho_historico = PASTA_HISTORICOS_MOBILENET / f"historico_mobilenetv2_{sufixo}.csv"
    return caminho_checkpoint, caminho_historico


def treinar_mobilenet_fold(
    base: pd.DataFrame,
    fold: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configurar_semente_torch(SEMENTE_ALEATORIA)
    PASTA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    PASTA_HISTORICOS_MOBILENET.mkdir(parents=True, exist_ok=True)

    df_treino = base.loc[fold["indices_treino"]].copy()
    df_validacao = base.loc[fold["indices_validacao"]].copy()
    df_teste = base.loc[fold["indices_teste"]].copy()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformacao_treino, transformacao_validacao = criar_transformacoes_mobilenet()
    dataset_treino = DatasetRecortes(df_treino, transformacao_treino)
    dataset_validacao = DatasetRecortes(df_validacao, transformacao_validacao)
    dataset_teste = DatasetRecortes(df_teste, transformacao_validacao)

    carregador_treino = criar_data_loader_mobilenet(dataset_treino, shuffle=True)
    carregador_validacao = criar_data_loader_mobilenet(dataset_validacao, shuffle=False)
    carregador_teste = criar_data_loader_mobilenet(dataset_teste, shuffle=False)

    modelo = criar_modelo_mobilenetv2().to(dispositivo)
    if dispositivo.type == "cuda" and USAR_CHANNELS_LAST_CUDA:
        modelo = modelo.to(memory_format=torch.channels_last)

    pesos_classes = calcular_pesos_classes_mobilenet(df_treino, dispositivo)
    criterio = nn.CrossEntropyLoss(weight=pesos_classes)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(USAR_MIXED_PRECISION and dispositivo.type == "cuda"),
    )

    caminho_checkpoint, caminho_historico = caminho_mobilenet_fold(fold)
    melhor_loss = float("inf")
    melhor_epoca = 0
    melhor_fase = None
    epocas_sem_melhora = 0
    fase_atual = None
    otimizador = None
    historico = []
    inicio = time.time()

    for epoca in range(1, EPOCHS_TOTAL + 1):
        fase, learning_rate = fase_para_epoca(epoca)
        if fase != fase_atual:
            fase_anterior = fase_atual
            fase_atual = fase
            if fase == "backbone_congelado":
                congelar_backbone(modelo)
            else:
                liberar_ultimos_blocos(modelo)
                if fase_anterior == "backbone_congelado":
                    epocas_sem_melhora = 0
            otimizador = criar_otimizador_mobilenet(modelo, learning_rate)

        metricas_treino = treinar_uma_epoca_mobilenet(
            modelo,
            carregador_treino,
            criterio,
            otimizador,
            dispositivo,
            scaler,
        )
        metricas_validacao = avaliar_mobilenet(
            modelo,
            carregador_validacao,
            criterio,
            dispositivo,
        )

        historico.append({
            "fold": int(fold["fold"]),
            "grupo_externo": fold["grupo_externo"],
            "grupo_validacao": fold["grupo_validacao"],
            "epoca": epoca,
            "fase": fase,
            "learning_rate": learning_rate,
            **{f"treino_{k}": v for k, v in metricas_treino.items()},
            **{f"validacao_{k}": v for k, v in metricas_validacao.items()},
        })

        loss_validacao = float(metricas_validacao["loss"])
        if loss_validacao < melhor_loss:
            melhor_loss = loss_validacao
            melhor_epoca = epoca
            melhor_fase = fase
            epocas_sem_melhora = 0
            checkpoint = {
                "modelo": "mobilenet_v2",
                "nome_modelo": "mobilenetv2_recortes",
                "protocolo": PROTOCOLO,
                "fold": int(fold["fold"]),
                "grupo_externo": fold["grupo_externo"],
                "grupo_validacao": fold["grupo_validacao"],
                "pesos_pre_treinados": PESOS_PRE_TREINADOS,
                "pesos_imagenet_carregados": PESOS_IMAGENET_CARREGADOS,
                "state_dict": modelo.state_dict(),
                "classes": CLASSES,
                "classe_positiva": CLASSE_POSITIVA,
                "class_to_idx": CLASSE_PARA_INDICE,
                "tamanho_imagem": TAMANHO_IMAGEM,
                "seed": SEMENTE_ALEATORIA,
                "epoca": epoca,
                "fase": fase,
                "melhor_loss_validacao": melhor_loss,
                "metricas_validacao": metricas_validacao,
                "data_treino": datetime.now().isoformat(timespec="seconds"),
            }
            torch.save(checkpoint, caminho_checkpoint)
        else:
            epocas_sem_melhora += 1

        if epocas_sem_melhora >= PACIENCIA_EARLY_STOPPING:
            break

    pd.DataFrame(historico).to_csv(caminho_historico, index=False, encoding="utf-8-sig")
    checkpoint = torch.load(caminho_checkpoint, map_location=dispositivo)
    modelo.load_state_dict(checkpoint["state_dict"])
    tempo_treino = time.time() - inicio

    prob_validacao = obter_probabilidades_mobilenet(modelo, carregador_validacao, dispositivo)
    prob_teste = obter_probabilidades_mobilenet(modelo, carregador_teste, dispositivo)
    contexto = completar_contexto_fold(contexto_modelo("mobilenetv2"), fold)

    parametros_json = json.dumps(
        {
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
            "checkpoint": str(caminho_checkpoint.relative_to(PASTA_PROJETO)),
            "historico": str(caminho_historico.relative_to(PASTA_PROJETO)),
            "melhor_fase": melhor_fase,
            "pesos_classes": pesos_classes.detach().cpu().tolist(),
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
        sort_keys=True,
    )

    resultado = avaliar_probabilidades(
        df_validacao,
        df_teste,
        prob_validacao,
        prob_teste,
        contexto,
        tempo_treino,
        melhor_epoca=melhor_epoca,
        melhor_loss_validacao=melhor_loss,
        parametros_json=parametros_json,
    )
    del modelo
    del otimizador
    del scaler
    del carregador_treino
    del carregador_validacao
    del carregador_teste
    del dataset_treino
    del dataset_validacao
    del dataset_teste
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return resultado
