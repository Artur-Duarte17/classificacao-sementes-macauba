from pathlib import Path
from datetime import datetime
import json
import random
import time
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ============================================================
# SCRIPT 24 - TREINAR MOBILENETV2 COM RECORTES
# ------------------------------------------------------------
# Objetivo:
# - Treinar MobileNetV2 com pesos ImageNet usando os recortes
# - Reusar a mesma divisao treino/validacao/teste do baseline
# - Salvar o melhor checkpoint por menor loss de validacao
#
# Este script treina modelo e deve ser executado manualmente no conda.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_DATASET = PASTA_PROJETO / "saidas" / "dataset_recortado"
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_DATASET_TABELAS = PASTA_TABELAS / "04_dataset_split"
PASTA_MODELO_TABELAS = PASTA_TABELAS / "06_modelos" / "mobilenetv2"
PASTA_MODELOS = PASTA_PROJETO / "saidas" / "modelos"

CLASSES = ["nao_contaminada", "contaminada"]
CLASSE_POSITIVA = "contaminada"
CLASSE_PARA_INDICE = {classe: indice for indice, classe in enumerate(CLASSES)}
INDICE_POSITIVO = CLASSE_PARA_INDICE[CLASSE_POSITIVA]

SEMENTE_ALEATORIA = 42
TAMANHO_IMAGEM = 224
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
USAR_MIXED_PRECISION = True
USAR_CHANNELS_LAST_CUDA = True
USAR_TF32_CUDA = True

EPOCHS_TOTAL = 40
EPOCHS_BACKBONE_CONGELADO = 5
PACIENCIA_EARLY_STOPPING = 8
LEARNING_RATE_CLASSIFICADOR = 1e-4
LEARNING_RATE_AJUSTE_FINO = 1e-5
WEIGHT_DECAY = 1e-4
BLOCOS_FINAIS_DESCONGELADOS = 4

NOME_MODELO = "mobilenetv2_recortes"
CAMINHO_MODELO = PASTA_MODELOS / f"{NOME_MODELO}_melhor.pt"
CAMINHO_CONFIG = PASTA_MODELOS / f"config_{NOME_MODELO}.json"
CAMINHO_HISTORICO = PASTA_MODELO_TABELAS / f"historico_treino_{NOME_MODELO}.csv"
CAMINHO_SPLIT = PASTA_DATASET_TABELAS / "divisao_treino_validacao_teste.csv"

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


class DatasetSementes(Dataset):
    def __init__(self, df: pd.DataFrame, transformacao):
        self.df = df.reset_index(drop=True)
        self.transformacao = transformacao

    def __len__(self):
        return len(self.df)

    def __getitem__(self, indice):
        linha = self.df.iloc[indice]
        caminho = PASTA_PROJETO / str(linha["caminho_imagem"])

        with Image.open(caminho) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

        imagem = self.transformacao(img)
        alvo = int(linha["alvo"])
        return imagem, alvo


def configurar_semente():
    random.seed(SEMENTE_ALEATORIA)
    np.random.seed(SEMENTE_ALEATORIA)
    torch.manual_seed(SEMENTE_ALEATORIA)
    torch.cuda.manual_seed_all(SEMENTE_ALEATORIA)


def carregar_divisao_recortes() -> pd.DataFrame:
    if not CAMINHO_SPLIT.exists():
        raise FileNotFoundError(
            "divisao_treino_validacao_teste.csv nao encontrado. "
            "Execute primeiro os scripts de preparacao do dataset."
        )

    df = pd.read_csv(CAMINHO_SPLIT)
    registros = []
    ausentes = []

    for _, linha in df.iterrows():
        classe = str(linha["classe"])
        nome_arquivo = str(linha["nome_arquivo"])
        caminho_recorte = PASTA_DATASET / classe / nome_arquivo

        if not caminho_recorte.exists():
            ausentes.append(str(caminho_recorte))
            continue

        registro = linha.to_dict()
        registro["caminho_imagem"] = str(caminho_recorte.relative_to(PASTA_PROJETO))
        registro["alvo"] = CLASSE_PARA_INDICE[classe]
        registros.append(registro)

    if ausentes:
        exemplos = "\n".join(ausentes[:10])
        raise FileNotFoundError(
            "Alguns recortes nao foram encontrados. Exemplos:\n"
            f"{exemplos}"
        )

    return pd.DataFrame(registros)


def criar_transformacoes():
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


def criar_modelo():
    try:
        pesos = models.MobileNet_V2_Weights.DEFAULT
        modelo = models.mobilenet_v2(weights=pesos)
        print("Pesos ImageNet da MobileNetV2 carregados.")
    except Exception as erro:
        print("AVISO: nao foi possivel carregar pesos ImageNet.")
        print(f"Motivo: {erro}")
        print("Continuando com pesos aleatorios.")
        modelo = models.mobilenet_v2(weights=None)

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


def criar_otimizador(modelo, learning_rate: float):
    return torch.optim.AdamW(
        parametros_treinaveis(modelo),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )


def calcular_pesos_classes(df_treino: pd.DataFrame, dispositivo):
    contagens = df_treino["alvo"].value_counts().reindex([0, 1], fill_value=0).astype(float)
    if (contagens == 0).any():
        raise ValueError(f"Treino sem uma das classes: {contagens.to_dict()}")
    total = float(contagens.sum())
    pesos = total / (len(CLASSES) * contagens)
    return torch.tensor(pesos.to_numpy(), dtype=torch.float32, device=dispositivo)


def calcular_metricas(alvos, predicoes) -> dict:
    precisao, recall, f1, _ = precision_recall_fscore_support(
        alvos,
        predicoes,
        labels=[INDICE_POSITIVO],
        average=None,
        zero_division=0,
    )

    matriz = confusion_matrix(alvos, predicoes, labels=[0, 1])
    tn, fp, fn, tp = matriz.ravel()
    sensibilidade = tp / max(tp + fn, 1)
    especificidade = tn / max(tn + fp, 1)

    return {
        "acuracia": accuracy_score(alvos, predicoes),
        "precisao_contaminada": float(precisao[0]),
        "recall_contaminada": float(recall[0]),
        "sensibilidade_contaminada": float(sensibilidade),
        "especificidade_nao_contaminada": float(especificidade),
        "f1_contaminada": float(f1[0]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def criar_data_loader(dataset, batch_size, shuffle):
    usar_workers = NUM_WORKERS > 0
    opcoes = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS and usar_workers,
    }
    if usar_workers:
        opcoes["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(dataset, **opcoes)


def configurar_desempenho_cuda(dispositivo):
    if dispositivo.type != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    if USAR_TF32_CUDA:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def transferir_imagens(imagens, dispositivo):
    usar_channels_last = dispositivo.type == "cuda" and USAR_CHANNELS_LAST_CUDA
    if usar_channels_last:
        return imagens.to(
            dispositivo,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
    return imagens.to(dispositivo, non_blocking=(dispositivo.type == "cuda"))


def treinar_uma_epoca(modelo, carregador, criterio, otimizador, dispositivo, scaler):
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

    metricas = calcular_metricas(alvos, predicoes)
    metricas["loss"] = sum(perdas) / max(len(perdas), 1)
    return metricas


@torch.no_grad()
def avaliar(modelo, carregador, criterio, dispositivo):
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

    metricas = calcular_metricas(alvos, predicoes)
    metricas["loss"] = sum(perdas) / max(len(perdas), 1)
    return metricas


def salvar_checkpoint(modelo, epoca, fase, metricas_validacao, melhor_loss):
    checkpoint = {
        "modelo": "mobilenet_v2",
        "nome_modelo": NOME_MODELO,
        "state_dict": modelo.state_dict(),
        "classes": CLASSES,
        "classe_positiva": CLASSE_POSITIVA,
        "class_to_idx": CLASSE_PARA_INDICE,
        "tamanho_imagem": TAMANHO_IMAGEM,
        "epoca": epoca,
        "fase": fase,
        "melhor_loss_validacao": melhor_loss,
        "metricas_validacao": metricas_validacao,
        "data_treino": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(checkpoint, CAMINHO_MODELO)


def fase_para_epoca(epoca: int) -> tuple[str, float]:
    if epoca <= EPOCHS_BACKBONE_CONGELADO:
        return "backbone_congelado", LEARNING_RATE_CLASSIFICADOR
    return "ajuste_fino_ultimos_blocos", LEARNING_RATE_AJUSTE_FINO


def main():
    print("=" * 60)
    print("TREINANDO MOBILENETV2 COM RECORTES")
    print("=" * 60)

    PASTA_MODELO_TABELAS.mkdir(parents=True, exist_ok=True)
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    configurar_semente()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configurar_desempenho_cuda(dispositivo)
    print(f"Dispositivo usado: {dispositivo}")
    print(
        "Config treino: "
        f"batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS}, "
        f"mixed_precision={USAR_MIXED_PRECISION}, pin_memory={PIN_MEMORY}, "
        f"persistent_workers={PERSISTENT_WORKERS}, entrada={TAMANHO_IMAGEM}x{TAMANHO_IMAGEM}"
    )
    if dispositivo.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("AVISO: CUDA nao foi detectado. O treino vai rodar na CPU.")

    df_split = carregar_divisao_recortes()
    print()
    print("Divisao reutilizada:")
    print(pd.crosstab(df_split["split"], df_split["classe"]).to_string())

    df_treino = df_split[df_split["split"] == "treino"].copy()
    df_validacao = df_split[df_split["split"] == "validacao"].copy()

    transformacao_treino, transformacao_validacao = criar_transformacoes()
    dataset_treino = DatasetSementes(df_treino, transformacao_treino)
    dataset_validacao = DatasetSementes(df_validacao, transformacao_validacao)
    carregador_treino = criar_data_loader(dataset_treino, BATCH_SIZE, shuffle=True)
    carregador_validacao = criar_data_loader(dataset_validacao, BATCH_SIZE, shuffle=False)

    modelo = criar_modelo().to(dispositivo)
    if dispositivo.type == "cuda" and USAR_CHANNELS_LAST_CUDA:
        modelo = modelo.to(memory_format=torch.channels_last)

    pesos_classes = calcular_pesos_classes(df_treino, dispositivo)
    criterio = nn.CrossEntropyLoss(weight=pesos_classes)
    scaler = torch.amp.GradScaler("cuda", enabled=(USAR_MIXED_PRECISION and dispositivo.type == "cuda"))

    melhor_loss = float("inf")
    melhor_epoca = 0
    epocas_sem_melhora = 0
    historico = []
    inicio = time.time()
    fase_atual = None
    otimizador = None

    for epoca in range(1, EPOCHS_TOTAL + 1):
        fase, learning_rate = fase_para_epoca(epoca)
        if fase != fase_atual:
            fase_atual = fase
            if fase == "backbone_congelado":
                congelar_backbone(modelo)
            else:
                liberar_ultimos_blocos(modelo)
            otimizador = criar_otimizador(modelo, learning_rate)
            print()
            print(f"Iniciando fase: {fase} | lr={learning_rate:g}")

        print()
        print(f"Epoca {epoca}/{EPOCHS_TOTAL}")
        metricas_treino = treinar_uma_epoca(
            modelo,
            carregador_treino,
            criterio,
            otimizador,
            dispositivo,
            scaler,
        )
        metricas_validacao = avaliar(modelo, carregador_validacao, criterio, dispositivo)

        linha = {
            "epoca": epoca,
            "fase": fase,
            "learning_rate": learning_rate,
            **{f"treino_{k}": v for k, v in metricas_treino.items()},
            **{f"validacao_{k}": v for k, v in metricas_validacao.items()},
        }
        historico.append(linha)

        print(
            "Treino: "
            f"loss={metricas_treino['loss']:.4f} "
            f"recall={metricas_treino['recall_contaminada']:.4f} "
            f"esp={metricas_treino['especificidade_nao_contaminada']:.4f}"
        )
        print(
            "Validacao: "
            f"loss={metricas_validacao['loss']:.4f} "
            f"recall={metricas_validacao['recall_contaminada']:.4f} "
            f"esp={metricas_validacao['especificidade_nao_contaminada']:.4f} "
            f"f1={metricas_validacao['f1_contaminada']:.4f}"
        )

        loss_validacao = float(metricas_validacao["loss"])
        if loss_validacao < melhor_loss:
            melhor_loss = loss_validacao
            melhor_epoca = epoca
            epocas_sem_melhora = 0
            salvar_checkpoint(modelo, epoca, fase, metricas_validacao, melhor_loss)
            print(f"Novo melhor modelo salvo em: {CAMINHO_MODELO}")
        else:
            epocas_sem_melhora += 1
            print(f"Sem melhora por {epocas_sem_melhora} epoca(s).")

        if epocas_sem_melhora >= PACIENCIA_EARLY_STOPPING:
            print("Early stopping acionado.")
            break

    pd.DataFrame(historico).to_csv(CAMINHO_HISTORICO, index=False, encoding="utf-8-sig")

    config = {
        "nome_modelo": NOME_MODELO,
        "arquitetura": "torchvision.models.mobilenet_v2",
        "pesos": "ImageNet DEFAULT quando disponivel",
        "dataset": "dataset_recortado",
        "split_reutilizado": str(CAMINHO_SPLIT),
        "classes": CLASSES,
        "classe_positiva": CLASSE_POSITIVA,
        "tamanho_imagem": TAMANHO_IMAGEM,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "mixed_precision": USAR_MIXED_PRECISION,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "epochs_total": EPOCHS_TOTAL,
        "epochs_backbone_congelado": EPOCHS_BACKBONE_CONGELADO,
        "learning_rate_classificador": LEARNING_RATE_CLASSIFICADOR,
        "learning_rate_ajuste_fino": LEARNING_RATE_AJUSTE_FINO,
        "weight_decay": WEIGHT_DECAY,
        "patiencia_early_stopping": PACIENCIA_EARLY_STOPPING,
        "blocos_finais_descongelados": BLOCOS_FINAIS_DESCONGELADOS,
        "pesos_classes": pesos_classes.detach().cpu().tolist(),
        "melhor_loss_validacao": melhor_loss,
        "melhor_epoca": melhor_epoca,
        "duracao_segundos": round(time.time() - inicio, 2),
    }
    CAMINHO_CONFIG.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("Treino MobileNetV2 concluido.")
    print(f"Historico salvo em: {CAMINHO_HISTORICO}")
    print(f"Config salvo em: {CAMINHO_CONFIG}")
    print(f"Melhor modelo salvo em: {CAMINHO_MODELO}")


if __name__ == "__main__":
    main()
