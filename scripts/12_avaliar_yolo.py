from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from ultralytics import YOLO


# ============================================================
# SCRIPT 12 - AVALIAR YOLO
# ------------------------------------------------------------
# Objetivo:
# - Rodar predicoes YOLO no conjunto de teste
# - Converter a melhor deteccao por imagem em classe predita
# - Comparar metricas com foco em recall de contaminada
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_YOLO = PASTA_PROJETO / "saidas" / "yolo_dataset"
PASTA_RUNS = PASTA_PROJETO / "saidas" / "yolo_runs"

NOME_RUN = "sementes_yolo_caixas_auto"
CAMINHO_MODELO = PASTA_RUNS / NOME_RUN / "weights" / "best.pt"
PASTA_TESTE = PASTA_YOLO / "images" / "test"

CLASSES = ["nao_contaminada", "contaminada"]
INDICE_POSITIVO = 1
CONF_MINIMA = 0.25
IMGSZ = 640
BATCH = 4


def carregar_reais() -> pd.DataFrame:
    caminho_relatorio = PASTA_TABELAS / "relatorio_dataset_yolo.csv"

    if not caminho_relatorio.exists():
        raise FileNotFoundError("relatorio_dataset_yolo.csv nao encontrado")

    df = pd.read_csv(caminho_relatorio)
    df = df[df["split_yolo"] == "test"].copy()
    return df[["imagem_yolo", "classe", "classe_yolo"]]


def predizer(modelo: YOLO, df_reais: pd.DataFrame, dispositivo) -> pd.DataFrame:
    imagens = df_reais["imagem_yolo"].tolist()

    resultados = modelo.predict(
        source=imagens,
        imgsz=IMGSZ,
        conf=CONF_MINIMA,
        device=dispositivo,
        batch=BATCH,
        verbose=False,
    )

    registros = []

    for linha, resultado in zip(df_reais.to_dict("records"), resultados):
        classe_predita = None
        confianca_predita = 0.0
        quantidade_deteccoes = 0

        if resultado.boxes is not None and len(resultado.boxes) > 0:
            quantidade_deteccoes = len(resultado.boxes)
            confs = resultado.boxes.conf.cpu().numpy()
            classes = resultado.boxes.cls.cpu().numpy().astype(int)
            melhor_indice = int(np.argmax(confs))
            classe_predita = int(classes[melhor_indice])
            confianca_predita = float(confs[melhor_indice])

        if classe_predita is None:
            # Se o YOLO nao detectar nada, conta como nao_contaminada por padrao.
            classe_predita = 0

        registros.append({
            "imagem_yolo": linha["imagem_yolo"],
            "classe_real": linha["classe"],
            "classe_real_yolo": int(linha["classe_yolo"]),
            "classe_predita_yolo": int(classe_predita),
            "classe_predita": CLASSES[int(classe_predita)],
            "confianca_predita": confianca_predita,
            "quantidade_deteccoes": quantidade_deteccoes,
        })

    return pd.DataFrame(registros)


def calcular_metricas(df: pd.DataFrame) -> dict:
    y_real = df["classe_real_yolo"].to_numpy()
    y_pred = df["classe_predita_yolo"].to_numpy()

    precisao, recall, f1, _ = precision_recall_fscore_support(
        y_real,
        y_pred,
        labels=[INDICE_POSITIVO],
        average=None,
        zero_division=0,
    )

    matriz = confusion_matrix(y_real, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matriz.ravel()

    return {
        "modelo": "yolo_caixas_automaticas",
        "acuracia": float(accuracy_score(y_real, y_pred)),
        "precisao_contaminada": float(precisao[0]),
        "recall_contaminada": float(recall[0]),
        "f1_contaminada": float(f1[0]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    print("=" * 60)
    print("AVALIANDO YOLO")
    print("=" * 60)

    if not CAMINHO_MODELO.exists():
        print("ERRO: modelo YOLO nao encontrado.")
        print(CAMINHO_MODELO)
        print("Execute primeiro: python scripts\\11_treinar_yolo.py")
        return

    if not PASTA_TESTE.exists():
        print("ERRO: pasta de teste YOLO nao encontrada.")
        print(PASTA_TESTE)
        return

    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo YOLO: {dispositivo}")

    modelo = YOLO(str(CAMINHO_MODELO))
    df_reais = carregar_reais()
    df_pred = predizer(modelo, df_reais, dispositivo)
    metricas = calcular_metricas(df_pred)

    caminho_predicoes = PASTA_TABELAS / "predicoes_yolo_teste.csv"
    caminho_metricas = PASTA_TABELAS / "metricas_yolo_teste.csv"

    df_pred.to_csv(caminho_predicoes, index=False, encoding="utf-8-sig")
    pd.DataFrame([metricas]).to_csv(caminho_metricas, index=False, encoding="utf-8-sig")

    print()
    print("Metricas no teste:")
    print(pd.DataFrame([metricas]).to_string(index=False))
    print()
    print("Arquivos gerados:")
    print(f"- {caminho_metricas}")
    print(f"- {caminho_predicoes}")


if __name__ == "__main__":
    main()
