from pathlib import Path
from datetime import datetime
import json
import time
import warnings

import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ============================================================
# SCRIPT 06 - TREINAR BASELINE
# ------------------------------------------------------------
# Objetivo:
# - Treinar um primeiro modelo simples com transferencia de aprendizado
# - Usar GPU automaticamente se PyTorch encontrar CUDA
# - Salvar o melhor modelo com foco no recall da classe contaminada
#
# Este script usa o dataset ja criado em saidas/dataset_binario.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_DATASET = PASTA_PROJETO / "saidas" / "dataset_binario"
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_MODELOS = PASTA_PROJETO / "saidas" / "modelos"

CLASSES = ["nao_contaminada", "contaminada"]
CLASSE_POSITIVA = "contaminada"
CLASSE_PARA_INDICE = {classe: indice for indice, classe in enumerate(CLASSES)}
INDICE_POSITIVO = CLASSE_PARA_INDICE[CLASSE_POSITIVA]

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SEMENTE_ALEATORIA = 42
TAMANHO_IMAGEM = 224
BATCH_SIZE = 8
EPOCHS = 12
LEARNING_RATE = 1e-4
PACIENCIA_EARLY_STOPPING = 4
NUM_WORKERS = 0

NOME_MODELO = "baseline_resnet18"
CAMINHO_MODELO = PASTA_MODELOS / f"{NOME_MODELO}_melhor.pt"
CAMINHO_HISTORICO = PASTA_TABELAS / f"historico_treino_{NOME_MODELO}.csv"
CAMINHO_SPLIT = PASTA_TABELAS / "divisao_treino_validacao_teste.csv"
CAMINHO_CONFIG = PASTA_MODELOS / f"config_{NOME_MODELO}.json"

# As imagens sao locais e fazem parte do experimento. Algumas fotos sao muito
# grandes e o Pillow emite esse aviso muitas vezes durante o treino.
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


def listar_imagens_dataset() -> pd.DataFrame:
    registros = []

    for classe in CLASSES:
        pasta_classe = PASTA_DATASET / classe

        if not pasta_classe.exists():
            raise FileNotFoundError(f"Pasta da classe nao encontrada: {pasta_classe}")

        for caminho in sorted(pasta_classe.iterdir()):
            if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_IMAGEM:
                registros.append({
                    "caminho_imagem": str(caminho.relative_to(PASTA_PROJETO)),
                    "nome_arquivo": caminho.name,
                    "classe": classe,
                    "alvo": CLASSE_PARA_INDICE[classe],
                })

    return pd.DataFrame(registros)


def criar_divisao_treino_validacao_teste(df: pd.DataFrame) -> pd.DataFrame:
    treino_validacao, teste = train_test_split(
        df,
        test_size=0.15,
        random_state=SEMENTE_ALEATORIA,
        stratify=df["classe"],
    )

    treino, validacao = train_test_split(
        treino_validacao,
        test_size=0.1765,
        random_state=SEMENTE_ALEATORIA,
        stratify=treino_validacao["classe"],
    )

    treino = treino.copy()
    validacao = validacao.copy()
    teste = teste.copy()

    treino["split"] = "treino"
    validacao["split"] = "validacao"
    teste["split"] = "teste"

    return pd.concat([treino, validacao, teste], ignore_index=True)


def criar_transformacoes():
    media = [0.485, 0.456, 0.406]
    desvio = [0.229, 0.224, 0.225]

    transformacao_treino = transforms.Compose([
        transforms.RandomResizedCrop(TAMANHO_IMAGEM, scale=(0.75, 1.0)),
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

    return {
        "acuracia": accuracy_score(alvos, predicoes),
        "precisao_contaminada": float(precisao[0]),
        "recall_contaminada": float(recall[0]),
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

    for imagens, y in carregador:
        imagens = imagens.to(dispositivo)
        y = y.to(dispositivo)

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

    for imagens, y in carregador:
        imagens = imagens.to(dispositivo)
        y = y.to(dispositivo)

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


def main():
    print("=" * 60)
    print("TREINANDO BASELINE - RESNET18")
    print("=" * 60)

    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEMENTE_ALEATORIA)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo usado: {dispositivo}")

    if dispositivo.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("AVISO: CUDA nao foi detectado. O treino vai rodar na CPU.")

    df = listar_imagens_dataset()

    if len(df) == 0:
        print("ERRO: nenhuma imagem encontrada no dataset binario.")
        return

    print()
    print("Imagens encontradas por classe:")
    print(df["classe"].value_counts().to_string())

    df_split = criar_divisao_treino_validacao_teste(df)
    df_split.to_csv(CAMINHO_SPLIT, index=False, encoding="utf-8-sig")

    print()
    print("Divisao criada:")
    print(pd.crosstab(df_split["split"], df_split["classe"]).to_string())
    print(f"Arquivo de divisao salvo em: {CAMINHO_SPLIT}")

    transformacao_treino, transformacao_validacao = criar_transformacoes()

    df_treino = df_split[df_split["split"] == "treino"].copy()
    df_validacao = df_split[df_split["split"] == "validacao"].copy()

    dataset_treino = DatasetSementes(df_treino, transformacao_treino)
    dataset_validacao = DatasetSementes(df_validacao, transformacao_validacao)

    carregador_treino = DataLoader(
        dataset_treino,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    carregador_validacao = DataLoader(
        dataset_validacao,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    modelo = criar_modelo().to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=dispositivo.type == "cuda")

    historico = []
    melhor_recall = -1.0
    melhor_f1 = -1.0
    epocas_sem_melhora = 0
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
        metricas_validacao = avaliar(
            modelo,
            carregador_validacao,
            criterio,
            dispositivo,
        )

        linha = {"epoca": epoca}
        for chave, valor in metricas_treino.items():
            linha[f"treino_{chave}"] = valor
        for chave, valor in metricas_validacao.items():
            linha[f"validacao_{chave}"] = valor
        historico.append(linha)

        print(
            "Treino     "
            f"loss={metricas_treino['loss']:.4f} "
            f"acc={metricas_treino['acuracia']:.3f} "
            f"recall_contaminada={metricas_treino['recall_contaminada']:.3f}"
        )
        print(
            "Validacao  "
            f"loss={metricas_validacao['loss']:.4f} "
            f"acc={metricas_validacao['acuracia']:.3f} "
            f"recall_contaminada={metricas_validacao['recall_contaminada']:.3f} "
            f"f1_contaminada={metricas_validacao['f1_contaminada']:.3f}"
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
        "classes": CLASSES,
        "classe_positiva": CLASSE_POSITIVA,
        "tamanho_imagem": TAMANHO_IMAGEM,
        "batch_size": BATCH_SIZE,
        "epochs_planejadas": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "semente_aleatoria": SEMENTE_ALEATORIA,
        "tempo_total_segundos": round(time.time() - inicio, 2),
    }
    CAMINHO_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print("TREINO CONCLUIDO")
    print("=" * 60)
    print(f"Melhor recall validacao contaminada: {melhor_recall:.3f}")
    print(f"Historico salvo em: {CAMINHO_HISTORICO}")
    print(f"Config salvo em: {CAMINHO_CONFIG}")
    print(f"Melhor modelo salvo em: {CAMINHO_MODELO}")


if __name__ == "__main__":
    main()
