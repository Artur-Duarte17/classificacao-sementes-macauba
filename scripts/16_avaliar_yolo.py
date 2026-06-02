from pathlib import Path
import os


# Evita conflito de OpenMP comum no Windows/conda ao misturar PyTorch,
# Ultralytics, OpenCV e bibliotecas numericas.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
from ultralytics import YOLO


# ============================================================
# SCRIPT 16 - AVALIAR YOLO
# ------------------------------------------------------------
# Objetivo:
# - Rodar predicoes YOLO na validacao e no teste
# - Avaliar thresholds usando apenas a validacao
# - Comparar metricas com foco em recall de contaminada
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_FIGURAS = PASTA_PROJETO / "saidas" / "figuras"
PASTA_YOLO = PASTA_PROJETO / "saidas" / "yolo_dataset"
PASTA_RUNS = PASTA_PROJETO / "saidas" / "yolo_runs"

NOME_RUN = "sementes_yolo_caixas_auto"
CAMINHO_MODELO = PASTA_RUNS / NOME_RUN / "weights" / "best.pt"

CLASSES = ["nao_contaminada", "contaminada"]
INDICE_POSITIVO = 1

# A predicao usa uma confianca baixa para capturar a melhor evidencia de cada
# classe. Os thresholds de decisao sao aplicados depois, em tabela.
CONF_PREDICAO = 0.05
CONF_REGRA_ATUAL = 0.25
RECALL_MINIMO_PRIORITARIO = 0.95

IMGSZ = 640
BATCH = 4

CAMINHO_PRED_VALIDACAO = PASTA_TABELAS / "predicoes_yolo_validacao.csv"
CAMINHO_PRED_TESTE = PASTA_TABELAS / "predicoes_yolo_teste.csv"
CAMINHO_THRESHOLDS = PASTA_TABELAS / "curva_threshold_yolo_validacao.csv"
CAMINHO_METRICAS = PASTA_TABELAS / "metricas_yolo_teste.csv"
CAMINHO_RESUMO_ORIGEM = PASTA_TABELAS / "resumo_yolo_por_origem_teste.csv"
CAMINHO_CURVA = PASTA_FIGURAS / "curva_threshold_yolo_validacao.png"
CAMINHO_MATRIZ = PASTA_FIGURAS / "matriz_confusao_yolo_teste.png"

CENARIO_PADRAO = "teste_threshold_prioridade_recall_validacao"


def carregar_reais(split_yolo: str) -> pd.DataFrame:
    caminho_relatorio = PASTA_TABELAS / "relatorio_dataset_yolo.csv"

    if not caminho_relatorio.exists():
        raise FileNotFoundError("relatorio_dataset_yolo.csv nao encontrado")

    df = pd.read_csv(caminho_relatorio)
    df = df[df["split_yolo"] == split_yolo].copy()
    return df[["imagem_yolo", "split_yolo", "classe", "classe_yolo"]]


def nome_origem(caminho_imagem: str) -> str:
    nome = Path(str(caminho_imagem)).name

    if nome.startswith("Micro-ondas__"):
        return "Micro-ondas"

    if nome.startswith("Piloto__"):
        return "Piloto"

    if nome.startswith("TESTE_2__"):
        return "TESTE_2"

    return "outros"


def nome_classe(indice: int) -> str:
    return CLASSES[int(indice)]


def predizer(modelo: YOLO, df_reais: pd.DataFrame, dispositivo, descricao: str) -> pd.DataFrame:
    registros = []
    linhas = df_reais.to_dict("records")

    for inicio in tqdm(range(0, len(linhas), BATCH), desc=descricao):
        lote = linhas[inicio:inicio + BATCH]
        imagens = [linha["imagem_yolo"] for linha in lote]

        resultados = modelo.predict(
            source=imagens,
            imgsz=IMGSZ,
            conf=CONF_PREDICAO,
            device=dispositivo,
            batch=BATCH,
            verbose=False,
        )

        for linha, resultado in zip(lote, resultados):
            registros.append(extrair_predicao(linha, resultado))

    return pd.DataFrame(registros)


def extrair_predicao(linha: dict, resultado) -> dict:
    conf_por_classe = {0: 0.0, 1: 0.0}
    classe_atual = 0
    confianca_atual = 0.0
    quantidade_deteccoes = 0
    quantidade_deteccoes_validas = 0

    if resultado.boxes is not None and len(resultado.boxes) > 0:
        confs = resultado.boxes.conf.cpu().numpy()
        classes = resultado.boxes.cls.cpu().numpy().astype(int)
        quantidade_deteccoes = len(confs)

        for classe, confianca in zip(classes, confs):
            if int(classe) in conf_por_classe:
                conf_por_classe[int(classe)] = max(
                    conf_por_classe[int(classe)],
                    float(confianca),
                )

        mascara_atual = confs >= CONF_REGRA_ATUAL
        quantidade_deteccoes_validas = int(mascara_atual.sum())

        if quantidade_deteccoes_validas > 0:
            indices_validos = np.where(mascara_atual)[0]
            melhor_indice = indices_validos[int(np.argmax(confs[indices_validos]))]
            classe_atual = int(classes[melhor_indice])
            confianca_atual = float(confs[melhor_indice])

    return {
        "imagem_yolo": linha["imagem_yolo"],
        "split_yolo": linha["split_yolo"],
        "origem": nome_origem(linha["imagem_yolo"]),
        "classe_real": linha["classe"],
        "classe_real_yolo": int(linha["classe_yolo"]),
        "conf_nao_contaminada": conf_por_classe[0],
        "conf_contaminada": conf_por_classe[1],
        "classe_predita_atual_yolo": int(classe_atual),
        "classe_predita_atual": nome_classe(classe_atual),
        "confianca_predita_atual": float(confianca_atual),
        "quantidade_deteccoes": int(quantidade_deteccoes),
        "quantidade_deteccoes_conf_atual": int(quantidade_deteccoes_validas),
    }


def predicao_por_threshold(df: pd.DataFrame, threshold: float) -> np.ndarray:
    return (df["conf_contaminada"].to_numpy() >= threshold).astype(int)


def confianca_por_predicao(df: pd.DataFrame, predicao: np.ndarray) -> np.ndarray:
    conf_nao = df["conf_nao_contaminada"].to_numpy()
    conf_cont = df["conf_contaminada"].to_numpy()
    return np.where(predicao == INDICE_POSITIVO, conf_cont, conf_nao)


def calcular_metricas(y_real, y_pred, cenario: str, threshold) -> dict:
    precisao, recall, f1, _ = precision_recall_fscore_support(
        y_real,
        y_pred,
        labels=[INDICE_POSITIVO],
        average=None,
        zero_division=0,
    )

    matriz = confusion_matrix(y_real, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matriz.ravel()
    sensibilidade = tp / max(tp + fn, 1)
    especificidade = tn / max(tn + fp, 1)

    return {
        "modelo": "yolo_caixas_automaticas",
        "cenario": cenario,
        "threshold": threshold,
        "acuracia": float(accuracy_score(y_real, y_pred)),
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
    y_real = df_validacao["classe_real_yolo"].to_numpy()
    registros = []

    for threshold in np.arange(0.05, 1.00, 0.05):
        threshold = round(float(threshold), 2)
        pred = predicao_por_threshold(df_validacao, threshold)
        registros.append(calcular_metricas(
            y_real,
            pred,
            cenario="validacao_threshold",
            threshold=threshold,
        ))

    return pd.DataFrame(registros)


def escolher_threshold_por_f1(df_thresholds: pd.DataFrame) -> float:
    melhor = df_thresholds.sort_values(
        ["f1_contaminada", "recall_contaminada", "acuracia"],
        ascending=[False, False, False],
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


def adicionar_predicoes_threshold(df: pd.DataFrame, nome: str, threshold: float) -> pd.DataFrame:
    pred = predicao_por_threshold(df, threshold)
    conf = confianca_por_predicao(df, pred)

    df[f"classe_predita_{nome}_yolo"] = pred
    df[f"classe_predita_{nome}"] = [nome_classe(indice) for indice in pred]
    df[f"confianca_predita_{nome}"] = conf
    return df


def definir_predicao_padrao(df: pd.DataFrame, nome: str, cenario: str) -> pd.DataFrame:
    df["cenario_predicao_padrao"] = cenario
    df["classe_predita_yolo"] = df[f"classe_predita_{nome}_yolo"]
    df["classe_predita"] = df[f"classe_predita_{nome}"]
    df["confianca_predita"] = df[f"confianca_predita_{nome}"]
    return df


def avaliar_teste(df_teste: pd.DataFrame, threshold_f1: float, threshold_recall: float) -> pd.DataFrame:
    y_real = df_teste["classe_real_yolo"].to_numpy()

    pred_atual = df_teste["classe_predita_atual_yolo"].to_numpy()
    pred_f1 = predicao_por_threshold(df_teste, threshold_f1)
    pred_recall = predicao_por_threshold(df_teste, threshold_recall)

    registros = [
        calcular_metricas(
            y_real,
            pred_atual,
            cenario="teste_regra_atual_melhor_deteccao_conf_0_25",
            threshold=CONF_REGRA_ATUAL,
        ),
        calcular_metricas(
            y_real,
            pred_f1,
            cenario="teste_threshold_melhor_f1_validacao",
            threshold=threshold_f1,
        ),
        calcular_metricas(
            y_real,
            pred_recall,
            cenario=CENARIO_PADRAO,
            threshold=threshold_recall,
        ),
    ]

    return pd.DataFrame(registros)


def resumo_por_origem(df_teste: pd.DataFrame) -> pd.DataFrame:
    registros = []

    for origem, grupo in df_teste.groupby("origem"):
        y_real = grupo["classe_real_yolo"].to_numpy()
        y_pred = grupo["classe_predita_yolo"].to_numpy()
        metricas = calcular_metricas(
            y_real,
            y_pred,
            cenario=CENARIO_PADRAO,
            threshold="threshold_prioridade_recall_validacao",
        )
        metricas["origem"] = origem
        metricas["quantidade"] = int(len(grupo))
        registros.append(metricas)

    colunas = [
        "origem",
        "quantidade",
        "acuracia",
        "precisao_contaminada",
        "recall_contaminada",
        "sensibilidade_contaminada",
        "especificidade_nao_contaminada",
        "f1_contaminada",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    return pd.DataFrame(registros)[colunas].sort_values("origem")


def plotar_curva_threshold(df_thresholds: pd.DataFrame, caminho_saida: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_thresholds["threshold"], df_thresholds["recall_contaminada"], label="Recall contaminada")
    ax.plot(df_thresholds["threshold"], df_thresholds["especificidade_nao_contaminada"], label="Especificidade nao contaminada")
    ax.plot(df_thresholds["threshold"], df_thresholds["precisao_contaminada"], label="Precisao contaminada")
    ax.plot(df_thresholds["threshold"], df_thresholds["f1_contaminada"], label="F1 contaminada")
    ax.set_title("YOLO - metricas por threshold na validacao")
    ax.set_xlabel("Threshold para classe contaminada")
    ax.set_ylabel("Metrica")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def plotar_matriz_confusao(metricas: dict, caminho_saida: Path):
    matriz = np.array([
        [metricas["tn"], metricas["fp"]],
        [metricas["fn"], metricas["tp"]],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matriz, cmap="Blues")
    ax.set_title("YOLO - matriz de confusao no teste")
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


def imprimir_comparacao_baseline(metricas_yolo: pd.DataFrame):
    caminho_baseline = PASTA_TABELAS / "metricas_baseline_resnet18_teste.csv"

    if not caminho_baseline.exists():
        return

    print()
    print("Comparacao com baseline ResNet18:")
    print(pd.read_csv(caminho_baseline).to_string(index=False))
    print()
    print("YOLO:")
    print(metricas_yolo.to_string(index=False))


def main():
    print("=" * 60)
    print("AVALIANDO YOLO")
    print("=" * 60)

    if not CAMINHO_MODELO.exists():
        print("ERRO: modelo YOLO nao encontrado.")
        print(CAMINHO_MODELO)
        print("Execute primeiro: python scripts\\15_treinar_yolo.py")
        return

    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo YOLO: {dispositivo}")

    modelo = YOLO(str(CAMINHO_MODELO))

    df_validacao = predizer(modelo, carregar_reais("val"), dispositivo, "Predizendo validacao")
    df_teste = predizer(modelo, carregar_reais("test"), dispositivo, "Predizendo teste")

    df_thresholds = gerar_curva_threshold(df_validacao)
    threshold_f1 = escolher_threshold_por_f1(df_thresholds)
    threshold_recall = escolher_threshold_por_recall(df_thresholds)

    df_validacao = adicionar_predicoes_threshold(df_validacao, "melhor_f1_validacao", threshold_f1)
    df_validacao = adicionar_predicoes_threshold(df_validacao, "prioridade_recall_validacao", threshold_recall)
    df_validacao = definir_predicao_padrao(
        df_validacao,
        "prioridade_recall_validacao",
        CENARIO_PADRAO,
    )

    df_teste = adicionar_predicoes_threshold(df_teste, "melhor_f1_validacao", threshold_f1)
    df_teste = adicionar_predicoes_threshold(df_teste, "prioridade_recall_validacao", threshold_recall)
    df_teste = definir_predicao_padrao(
        df_teste,
        "prioridade_recall_validacao",
        CENARIO_PADRAO,
    )

    metricas_teste = avaliar_teste(df_teste, threshold_f1, threshold_recall)
    resumo_origem = resumo_por_origem(df_teste)

    df_validacao.to_csv(CAMINHO_PRED_VALIDACAO, index=False, encoding="utf-8-sig")
    df_teste.to_csv(CAMINHO_PRED_TESTE, index=False, encoding="utf-8-sig")
    df_thresholds.to_csv(CAMINHO_THRESHOLDS, index=False, encoding="utf-8-sig")
    metricas_teste.to_csv(CAMINHO_METRICAS, index=False, encoding="utf-8-sig")
    resumo_origem.to_csv(CAMINHO_RESUMO_ORIGEM, index=False, encoding="utf-8-sig")

    plotar_curva_threshold(df_thresholds, CAMINHO_CURVA)
    metricas_padrao = metricas_teste[metricas_teste["cenario"] == CENARIO_PADRAO].iloc[0].to_dict()
    plotar_matriz_confusao(metricas_padrao, CAMINHO_MATRIZ)

    print()
    print("Threshold escolhido por F1 na validacao:")
    print(threshold_f1)
    print("Threshold escolhido por prioridade de recall na validacao:")
    print(threshold_recall)
    imprimir_comparacao_baseline(metricas_teste)

    print()
    print("Resumo por origem no teste:")
    print(resumo_origem.to_string(index=False))
    print()
    print("Arquivos gerados:")
    for caminho in [
        CAMINHO_PRED_VALIDACAO,
        CAMINHO_PRED_TESTE,
        CAMINHO_THRESHOLDS,
        CAMINHO_METRICAS,
        CAMINHO_RESUMO_ORIGEM,
        CAMINHO_CURVA,
        CAMINHO_MATRIZ,
    ]:
        print(f"- {caminho}")
    print()
    print("Proximo passo:")
    print("python scripts\\17_conferir_erros_yolo.py")


if __name__ == "__main__":
    main()
