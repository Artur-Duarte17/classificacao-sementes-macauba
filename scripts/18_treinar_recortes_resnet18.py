from pathlib import Path
from datetime import datetime
import json
import time
import warnings

import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ============================================================
# SCRIPT 18 - TREINAR RESNET18 COM RECORTES
# ------------------------------------------------------------
# Objetivo:
# - Treinar um classificador usando somente os recortes da semente
# - Reusar a mesma divisao treino/validacao/teste do baseline
# - Comparar se remover fundo/regua/pinca/etiqueta ajuda
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_DATASET = PASTA_PROJETO / "saidas" / "dataset_recortado"
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_MODELOS = PASTA_PROJETO / "saidas" / "modelos"

CLASSES = ["nao_contaminada", "contaminada"]
CLASSE_POSITIVA = "contaminada"
CLASSE_PARA_INDICE = {classe: indice for indice, classe in enumerate(CLASSES)}
INDICE_POSITIVO = CLASSE_PARA_INDICE[CLASSE_POSITIVA]

SEMENTE_ALEATORIA = 42
TAMANHO_IMAGEM = 224
BATCH_SIZE = 24
EPOCHS = 12
LEARNING_RATE = 1e-4
PACIENCIA_EARLY_STOPPING = 4
NUM_WORKERS = 4
PIN_MEMORY_CUDA = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
USAR_CHANNELS_LAST_CUDA = True
USAR_TF32_CUDA = True

NOME_MODELO = "recortes_resnet18"
CAMINHO_MODELO = PASTA_MODELOS / f"{NOME_MODELO}_melhor.pt"
CAMINHO_HISTORICO = PASTA_TABELAS / f"historico_treino_{NOME_MODELO}.csv"
CAMINHO_SPLIT = PASTA_TABELAS / "divisao_treino_validacao_teste.csv"
CAMINHO_CONFIG = PASTA_MODELOS / f"config_{NOME_MODELO}.json"

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


def carregar_divisao_recortes() -> pd.DataFrame:
    if not CAMINHO_SPLIT.exists():
        raise FileNotFoundError(
            "divisao_treino_validacao_teste.csv nao encontrado. "
            "Execute primeiro: python scripts\\06_treinar_baseline.py"
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
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
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
        pesos = models.ResNet18_Weights.DEFAULT
        modelo = models.resnet18(weights=pesos)
        print("Pesos pre-treinados da ResNet18 carregados.")
    except Exception as erro:
        print("AVISO: nao foi possivel carregar pesos pre-treinados.")
        print(f"Motivo: {erro}")
        print("Continuando com pesos aleatorios.")
        modelo = models.resnet18(weights=None)

    modelo.fc = nn.Linear(modelo.fc.in_features, len(CLASSES))
    return modelo


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


def treinar_uma_epoca(modelo, carregador, criterio, otimizador, dispositivo, scaler):
    modelo.train()
    perdas = []
    alvos = []
    predicoes = []
    usar_amp = dispositivo.type == "cuda"
    transferencia_assincrona = dispositivo.type == "cuda"
    usar_channels_last = dispositivo.type == "cuda" and USAR_CHANNELS_LAST_CUDA

    for imagens, y in carregador:
        if usar_channels_last:
            imagens = imagens.to(
                dispositivo,
                non_blocking=transferencia_assincrona,
                memory_format=torch.channels_last,
            )
        else:
            imagens = imagens.to(dispositivo, non_blocking=transferencia_assincrona)
        y = y.to(dispositivo, non_blocking=transferencia_assincrona)

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
    transferencia_assincrona = dispositivo.type == "cuda"
    usar_channels_last = dispositivo.type == "cuda" and USAR_CHANNELS_LAST_CUDA

    for imagens, y in carregador:
        if usar_channels_last:
            imagens = imagens.to(
                dispositivo,
                non_blocking=transferencia_assincrona,
                memory_format=torch.channels_last,
            )
        else:
            imagens = imagens.to(dispositivo, non_blocking=transferencia_assincrona)
        y = y.to(dispositivo, non_blocking=transferencia_assincrona)

        saidas = modelo(imagens)
        perda = criterio(saidas, y)

        perdas.append(float(perda.item()))
        pred = torch.argmax(saidas, dim=1)
        alvos.extend(y.cpu().tolist())
        predicoes.extend(pred.cpu().tolist())

    metricas = calcular_metricas(alvos, predicoes)
    metricas["loss"] = sum(perdas) / max(len(perdas), 1)
    return metricas


def salvar_checkpoint(modelo, epoca, metricas_validacao):
    checkpoint = {
        "modelo": "resnet18",
        "nome_modelo": NOME_MODELO,
        "state_dict": modelo.state_dict(),
        "classes": CLASSES,
        "classe_positiva": CLASSE_POSITIVA,
        "class_to_idx": CLASSE_PARA_INDICE,
        "tamanho_imagem": TAMANHO_IMAGEM,
        "epoca": epoca,
        "metricas_validacao": metricas_validacao,
        "data_treino": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(checkpoint, CAMINHO_MODELO)


def criar_data_loader(dataset, batch_size, shuffle, dispositivo):
    usar_workers = NUM_WORKERS > 0
    usar_pin_memory = PIN_MEMORY_CUDA and dispositivo.type == "cuda"

    opcoes = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": usar_pin_memory,
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


def main():
    print("=" * 60)
    print("TREINANDO RESNET18 COM RECORTES")
    print("=" * 60)

    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEMENTE_ALEATORIA)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configurar_desempenho_cuda(dispositivo)
    print(f"Dispositivo usado: {dispositivo}")

    if dispositivo.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            "Desempenho CUDA: "
            f"batch={BATCH_SIZE}, workers={NUM_WORKERS}, "
            f"pin_memory={PIN_MEMORY_CUDA}, prefetch={PREFETCH_FACTOR}, "
            f"channels_last={USAR_CHANNELS_LAST_CUDA}, tf32={USAR_TF32_CUDA}"
        )
    else:
        print("AVISO: CUDA nao foi detectado. O treino vai rodar na CPU.")

    df_split = carregar_divisao_recortes()

    print()
    print("Recortes encontrados por classe:")
    print(df_split["classe"].value_counts().to_string())
    print()
    print("Divisao reutilizada:")
    print(pd.crosstab(df_split["split"], df_split["classe"]).to_string())

    transformacao_treino, transformacao_validacao = criar_transformacoes()

    df_treino = df_split[df_split["split"] == "treino"].copy()
    df_validacao = df_split[df_split["split"] == "validacao"].copy()

    dataset_treino = DatasetSementes(df_treino, transformacao_treino)
    dataset_validacao = DatasetSementes(df_validacao, transformacao_validacao)

    carregador_treino = criar_data_loader(
        dataset_treino,
        batch_size=BATCH_SIZE,
        shuffle=True,
        dispositivo=dispositivo,
    )
    carregador_validacao = criar_data_loader(
        dataset_validacao,
        batch_size=BATCH_SIZE,
        shuffle=False,
        dispositivo=dispositivo,
    )

    modelo = criar_modelo().to(dispositivo)
    if dispositivo.type == "cuda" and USAR_CHANNELS_LAST_CUDA:
        modelo = modelo.to(memory_format=torch.channels_last)

    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=dispositivo.type == "cuda")

    melhor_recall = -1
    melhor_f1 = -1
    epocas_sem_melhora = 0
    historico = []
    inicio = time.time()

    for epoca in range(1, EPOCHS + 1):
        print()
        print(f"Epoca {epoca}/{EPOCHS}")

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

        recall_atual = metricas_validacao["recall_contaminada"]
        f1_atual = metricas_validacao["f1_contaminada"]
        melhorou = (recall_atual > melhor_recall) or (
            recall_atual == melhor_recall and f1_atual > melhor_f1
        )

        if melhorou:
            melhor_recall = recall_atual
            melhor_f1 = f1_atual
            epocas_sem_melhora = 0
            salvar_checkpoint(modelo, epoca, metricas_validacao)
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
        "dataset": "dataset_recortado",
        "split_reutilizado": str(CAMINHO_SPLIT),
        "classes": CLASSES,
        "classe_positiva": CLASSE_POSITIVA,
        "tamanho_imagem": TAMANHO_IMAGEM,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "num_workers": NUM_WORKERS,
        "pin_memory_cuda": PIN_MEMORY_CUDA,
        "persistent_workers": PERSISTENT_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "channels_last_cuda": USAR_CHANNELS_LAST_CUDA,
        "tf32_cuda": USAR_TF32_CUDA,
        "duracao_segundos": round(time.time() - inicio, 2),
        "melhor_recall_validacao": melhor_recall,
        "melhor_f1_validacao": melhor_f1,
    }
    CAMINHO_CONFIG.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("Treino com recortes concluido.")
    print(f"Historico salvo em: {CAMINHO_HISTORICO}")
    print(f"Config salvo em: {CAMINHO_CONFIG}")
    print(f"Melhor modelo salvo em: {CAMINHO_MODELO}")


if __name__ == "__main__":
    main()
