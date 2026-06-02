from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ============================================================
# SCRIPT 19 - AVALIAR RESNET18 COM RECORTES
# ------------------------------------------------------------
# Objetivo:
# - Avaliar o classificador treinado nos recortes da semente
# - Gerar curva de threshold, matriz e metricas no teste
# - Comparar com baseline imagem inteira e YOLO
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_DATASET = PASTA_PROJETO / "saidas" / "dataset_recortado"
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_MODELOS = PASTA_PROJETO / "saidas" / "modelos"
PASTA_FIGURAS = PASTA_PROJETO / "saidas" / "figuras"

CLASSES = ["nao_contaminada", "contaminada"]
CLASSE_POSITIVA = "contaminada"
CLASSE_PARA_INDICE = {classe: indice for indice, classe in enumerate(CLASSES)}
INDICE_POSITIVO = CLASSE_PARA_INDICE[CLASSE_POSITIVA]

TAMANHO_IMAGEM = 224
BATCH_SIZE = 8
NUM_WORKERS = 0
NOME_MODELO = "recortes_resnet18"

CAMINHO_MODELO = PASTA_MODELOS / f"{NOME_MODELO}_melhor.pt"
CAMINHO_SPLIT = PASTA_TABELAS / "divisao_treino_validacao_teste.csv"
CAMINHO_METRICAS = PASTA_TABELAS / f"metricas_{NOME_MODELO}_teste.csv"
CAMINHO_PREDICOES = PASTA_TABELAS / f"predicoes_{NOME_MODELO}_teste.csv"
CAMINHO_THRESHOLDS = PASTA_TABELAS / f"curva_threshold_{NOME_MODELO}_validacao.csv"
CAMINHO_MATRIZ = PASTA_FIGURAS / f"matriz_confusao_{NOME_MODELO}_teste.png"
CAMINHO_CURVA = PASTA_FIGURAS / f"curva_threshold_{NOME_MODELO}_validacao.png"

RECALL_MINIMO_PRIORITARIO = 0.95

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

        return imagem, alvo, str(linha["caminho_imagem"]), str(linha["classe"])


def carregar_divisao_recortes() -> pd.DataFrame:
    if not CAMINHO_SPLIT.exists():
        raise FileNotFoundError("divisao_treino_validacao_teste.csv nao encontrado")

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


def criar_transformacao():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(TAMANHO_IMAGEM),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def criar_modelo():
    modelo = models.resnet18(weights=None)
    modelo.fc = nn.Linear(modelo.fc.in_features, len(CLASSES))
    return modelo


@torch.no_grad()
def obter_predicoes(modelo, carregador, dispositivo) -> pd.DataFrame:
    modelo.eval()
    registros = []

    for imagens, alvos, caminhos, classes_reais in carregador:
        imagens = imagens.to(dispositivo)
        saidas = modelo(imagens)
        probabilidades = torch.softmax(saidas, dim=1)[:, INDICE_POSITIVO]

        for alvo, caminho, classe_real, prob in zip(
            alvos.tolist(),
            caminhos,
            classes_reais,
            probabilidades.cpu().tolist(),
        ):
            registros.append({
                "caminho_imagem": caminho,
                "classe_real": classe_real,
                "alvo": int(alvo),
                "prob_contaminada": float(prob),
            })

    return pd.DataFrame(registros)


def calcular_metricas(y_real, prob_contaminada, threshold: float) -> dict:
    pred = (prob_contaminada >= threshold).astype(int)

    precisao, recall, f1, _ = precision_recall_fscore_support(
        y_real,
        pred,
        labels=[INDICE_POSITIVO],
        average=None,
        zero_division=0,
    )

    matriz = confusion_matrix(y_real, pred, labels=[0, 1])
    tn, fp, fn, tp = matriz.ravel()
    sensibilidade = tp / max(tp + fn, 1)
    especificidade = tn / max(tn + fp, 1)

    return {
        "threshold": float(threshold),
        "acuracia": float(accuracy_score(y_real, pred)),
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


def gerar_curva_threshold(df_validacao: pd.DataFrame) -> pd.DataFrame:
    y_real = df_validacao["alvo"].to_numpy()
    prob = df_validacao["prob_contaminada"].to_numpy()
    registros = []

    for threshold in np.arange(0.05, 1.00, 0.05):
        registros.append(calcular_metricas(y_real, prob, round(float(threshold), 2)))

    return pd.DataFrame(registros)


def plotar_matriz_confusao(metricas: dict, caminho_saida: Path):
    matriz = np.array([
        [metricas["tn"], metricas["fp"]],
        [metricas["fn"], metricas["tp"]],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matriz, cmap="Blues")
    ax.set_title("Recortes ResNet18 - matriz de confusao no teste")
    ax.set_xticks([0, 1], labels=CLASSES)
    ax.set_yticks([0, 1], labels=CLASSES)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center", fontsize=16)

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def plotar_curva_threshold(df_thresholds: pd.DataFrame, caminho_saida: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_thresholds["threshold"], df_thresholds["recall_contaminada"], label="Recall contaminada")
    ax.plot(df_thresholds["threshold"], df_thresholds["especificidade_nao_contaminada"], label="Especificidade nao contaminada")
    ax.plot(df_thresholds["threshold"], df_thresholds["precisao_contaminada"], label="Precisao contaminada")
    ax.plot(df_thresholds["threshold"], df_thresholds["f1_contaminada"], label="F1 contaminada")
    ax.set_title("Recortes ResNet18 - metricas por threshold na validacao")
    ax.set_xlabel("Threshold para classe contaminada")
    ax.set_ylabel("Metrica")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def escolher_threshold_por_f1(df_thresholds: pd.DataFrame) -> float:
    melhor = df_thresholds.sort_values(
        ["f1_contaminada", "recall_contaminada"],
        ascending=[False, False],
    ).iloc[0]
    return float(melhor["threshold"])


def escolher_threshold_por_recall(df_thresholds: pd.DataFrame) -> float:
    candidatos = df_thresholds[
        df_thresholds["recall_contaminada"] >= RECALL_MINIMO_PRIORITARIO
    ].copy()

    if len(candidatos) == 0:
        candidatos = df_thresholds.copy()
        ordenacao = ["recall_contaminada", "f1_contaminada", "precisao_contaminada"]
    else:
        ordenacao = ["f1_contaminada", "recall_contaminada", "precisao_contaminada"]

    melhor = candidatos.sort_values(ordenacao, ascending=[False, False, False]).iloc[0]
    return float(melhor["threshold"])


def main():
    print("=" * 60)
    print("AVALIANDO RESNET18 COM RECORTES")
    print("=" * 60)

    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

    if not CAMINHO_MODELO.exists():
        print("ERRO: modelo nao encontrado.")
        print(CAMINHO_MODELO)
        print("Execute primeiro: python scripts\\18_treinar_recortes_resnet18.py")
        return

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo usado: {dispositivo}")

    df_split = carregar_divisao_recortes()
    transformacao = criar_transformacao()

    checkpoint = torch.load(CAMINHO_MODELO, map_location=dispositivo)
    modelo = criar_modelo().to(dispositivo)
    modelo.load_state_dict(checkpoint["state_dict"])

    predicoes_por_split = {}

    for split in ["validacao", "teste"]:
        df_split_atual = df_split[df_split["split"] == split].copy()
        dataset = DatasetSementes(df_split_atual, transformacao)
        carregador = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        predicoes_por_split[split] = obter_predicoes(modelo, carregador, dispositivo)

    df_thresholds = gerar_curva_threshold(predicoes_por_split["validacao"])
    df_thresholds.to_csv(CAMINHO_THRESHOLDS, index=False, encoding="utf-8-sig")
    plotar_curva_threshold(df_thresholds, CAMINHO_CURVA)

    threshold_f1 = escolher_threshold_por_f1(df_thresholds)
    threshold_recall = escolher_threshold_por_recall(df_thresholds)

    df_teste = predicoes_por_split["teste"].copy()
    prob_teste = df_teste["prob_contaminada"].to_numpy()
    y_teste = df_teste["alvo"].to_numpy()

    metricas_05 = calcular_metricas(y_teste, prob_teste, threshold=0.50)
    metricas_f1 = calcular_metricas(y_teste, prob_teste, threshold=threshold_f1)
    metricas_recall = calcular_metricas(y_teste, prob_teste, threshold=threshold_recall)

    resultados = {
        "teste_threshold_0_50": metricas_05,
        "teste_threshold_melhor_f1_validacao": metricas_f1,
        "teste_threshold_prioridade_recall_validacao": metricas_recall,
    }

    df_metricas = pd.DataFrame([
        {"cenario": nome, **metricas}
        for nome, metricas in resultados.items()
    ])
    df_metricas.to_csv(CAMINHO_METRICAS, index=False, encoding="utf-8-sig")

    df_teste["predito_threshold_0_50"] = np.where(
        df_teste["prob_contaminada"] >= 0.50,
        "contaminada",
        "nao_contaminada",
    )
    df_teste["predito_threshold_melhor_f1_validacao"] = np.where(
        df_teste["prob_contaminada"] >= threshold_f1,
        "contaminada",
        "nao_contaminada",
    )
    df_teste["predito_threshold_prioridade_recall_validacao"] = np.where(
        df_teste["prob_contaminada"] >= threshold_recall,
        "contaminada",
        "nao_contaminada",
    )
    df_teste.to_csv(CAMINHO_PREDICOES, index=False, encoding="utf-8-sig")

    plotar_matriz_confusao(metricas_05, CAMINHO_MATRIZ)

    print()
    print("Metricas no teste:")
    print(df_metricas.to_string(index=False))
    print()
    print("Arquivos gerados:")
    print(f"- {CAMINHO_METRICAS}")
    print(f"- {CAMINHO_PREDICOES}")
    print(f"- {CAMINHO_THRESHOLDS}")
    print(f"- {CAMINHO_MATRIZ}")
    print(f"- {CAMINHO_CURVA}")


if __name__ == "__main__":
    main()
