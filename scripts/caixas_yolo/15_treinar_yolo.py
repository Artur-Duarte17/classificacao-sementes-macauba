from pathlib import Path
import os


# Evita conflito de OpenMP comum no Windows/conda ao misturar PyTorch,
# Ultralytics, OpenCV e bibliotecas numericas.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from ultralytics import YOLO


# ============================================================
# SCRIPT 15 - TREINAR YOLO
# ------------------------------------------------------------
# Objetivo:
# - Treinar YOLO detect usando as caixas automaticas
# - Usar GPU automaticamente se CUDA estiver disponivel
#
# Execute apenas depois de conferir as caixas automaticas.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_YOLO = PASTA_PROJETO / "saidas" / "yolo_dataset"
PASTA_RUNS = PASTA_PROJETO / "saidas" / "yolo_runs"

DATA_YAML = PASTA_YOLO / "data.yaml"
MODELO_BASE = "yolo11n.pt" # trocar depois yolo11s.pt ou yolo11m.pt ou yolo11l.pt
EPOCHS = 40
IMGSZ = 640
BATCH = 6
PACIENCIA = 8
NOME_RUN = "sementes_yolo_caixas_auto"


def main():
    print("=" * 60)
    print("TREINANDO YOLO COM CAIXAS AUTOMATICAS")
    print("=" * 60)

    if not DATA_YAML.exists():
        print("ERRO: data.yaml nao encontrado.")
        print(DATA_YAML)
        print("Execute primeiro: python scripts\\14_criar_dataset_yolo.py")
        return

    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo YOLO: {dispositivo}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("AVISO: CUDA nao detectado. O treino vai rodar na CPU.")

    modelo = YOLO(MODELO_BASE)

    resultado = modelo.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PACIENCIA,
        device=dispositivo,
        project=str(PASTA_RUNS),
        name=NOME_RUN,
        exist_ok=True,
        workers=0,
    )

    print()
    print("Treino YOLO concluido.")
    print(resultado)
    print()


if __name__ == "__main__":
    main()



