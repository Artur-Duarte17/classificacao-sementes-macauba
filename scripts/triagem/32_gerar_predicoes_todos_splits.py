from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ============================================================
# SCRIPT 32 - GERAR PREDICOES PARA TODOS OS SPLITS
# ------------------------------------------------------------
# Objetivo:
# - Usar os modelos ja treinados
# - Gerar probabilidades para treino, validacao e teste
# - Preparar base para calibrar thresholds na validacao
#
# Este script nao treina modelos e nao altera imagens.
# Ele deve ser executado manualmente pelo usuario.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_DATASET_TABELAS = PASTA_TABELAS / "04_dataset_split"
PASTA_TRIAGEM_TABELAS = PASTA_TABELAS / "08_triagem"
PASTA_MODELOS = PASTA_PROJETO / "saidas" / "modelos"
PASTA_DATASET_RECORTADO = PASTA_PROJETO / "saidas" / "dataset_recortado"

CAMINHO_SPLIT = PASTA_DATASET_TABELAS / "divisao_treino_validacao_teste.csv"
CAMINHO_RELATORIO_COPIA = PASTA_DATASET_TABELAS / "relatorio_copia_dataset_binario.csv"
CAMINHO_MODELO_BASELINE = PASTA_MODELOS / "baseline_resnet18_melhor.pt"
CAMINHO_MODELO_RECORTES = PASTA_MODELOS / "recortes_resnet18_melhor.pt"

CAMINHO_PREDICOES_TODOS_SPLITS = (
    PASTA_TRIAGEM_TABELAS / "predicoes_todos_splits.csv"
)
CAMINHO_RESUMO_PREDICOES = PASTA_TRIAGEM_TABELAS / "resumo_predicoes_todos_splits.csv"

CLASSES = ["nao_contaminada", "contaminada"]
CLASSE_POSITIVA = "contaminada"
CLASSE_PARA_INDICE = {classe: indice for indice, classe in enumerate(CLASSES)}
INDICE_POSITIVO = CLASSE_PARA_INDICE[CLASSE_POSITIVA]

TAMANHO_IMAGEM = 224
BATCH_SIZE = 8
NUM_WORKERS = 0
LIMIAR_ALTO_RISCO = 0.70
LIMIAR_BAIXO_RISCO = 0.30
EXIGIR_RECORTES = True

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


def resolver_caminho_imagem(caminho_texto: str) -> Path:
    caminho = Path(str(caminho_texto))
    if caminho.is_absolute():
        return caminho
    return PASTA_PROJETO / caminho


class DatasetSementes(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transformacao,
        coluna_caminho: str,
        coluna_modelo: str,
    ):
        self.df = df.reset_index(drop=True)
        self.transformacao = transformacao
        self.coluna_caminho = coluna_caminho
        self.coluna_modelo = coluna_modelo

    def __len__(self):
        return len(self.df)

    def __getitem__(self, indice):
        linha = self.df.iloc[indice]
        caminho = resolver_caminho_imagem(str(linha[self.coluna_caminho]))

        with Image.open(caminho) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

        imagem = self.transformacao(img)

        return {
            "imagem": imagem,
            "nome_arquivo": str(linha["nome_arquivo"]),
            "split": str(linha["split"]),
            "classe_real": str(linha["classe"]),
            "alvo": int(linha["alvo"]),
            self.coluna_modelo: str(linha[self.coluna_caminho]),
        }


def ler_csv_obrigatorio(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo obrigatorio nao encontrado: {caminho}")
    return pd.read_csv(caminho)


def validar_arquivo(caminho: Path, descricao: str):
    if not caminho.exists():
        raise FileNotFoundError(f"{descricao} nao encontrado: {caminho}")


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


def carregar_modelo(caminho_modelo: Path, dispositivo):
    validar_arquivo(caminho_modelo, "Modelo treinado")
    checkpoint = torch.load(caminho_modelo, map_location=dispositivo)
    modelo = criar_modelo().to(dispositivo)
    modelo.load_state_dict(checkpoint["state_dict"])
    return modelo


def preparar_split_baseline(df_split: pd.DataFrame) -> pd.DataFrame:
    df = df_split.copy()
    df["caminho_imagem_baseline"] = df["caminho_imagem"].astype(str)

    caminhos_existentes = df["caminho_imagem_baseline"].map(
        lambda caminho: resolver_caminho_imagem(caminho).exists()
    )
    if caminhos_existentes.all():
        return df

    if not CAMINHO_RELATORIO_COPIA.exists():
        faltantes = df.loc[~caminhos_existentes, "caminho_imagem_baseline"].head(10)
        exemplos = "\n".join(f"- {caminho}" for caminho in faltantes)
        raise FileNotFoundError(
            "Algumas imagens do dataset_binario nao foram encontradas e "
            "relatorio_copia_dataset_binario.csv tambem nao existe para "
            "mapear os caminhos originais. Exemplos:\n"
            f"{exemplos}"
        )

    relatorio = pd.read_csv(CAMINHO_RELATORIO_COPIA)
    colunas_necessarias = ["nome_copiado", "caminho_original", "status_copia"]
    faltantes = [
        coluna for coluna in colunas_necessarias if coluna not in relatorio.columns
    ]
    if faltantes:
        raise ValueError(
            "relatorio_copia_dataset_binario.csv esta incompleto. "
            f"Colunas ausentes: {faltantes}"
        )

    mapa_original = (
        relatorio[relatorio["status_copia"] == "ok"]
        .drop_duplicates("nome_copiado")
        .set_index("nome_copiado")["caminho_original"]
    )
    df.loc[~caminhos_existentes, "caminho_imagem_baseline"] = df.loc[
        ~caminhos_existentes, "nome_arquivo"
    ].map(mapa_original)

    ainda_faltantes = df["caminho_imagem_baseline"].isna() | ~df[
        "caminho_imagem_baseline"
    ].map(lambda caminho: resolver_caminho_imagem(caminho).exists())
    if ainda_faltantes.any():
        exemplos = df.loc[ainda_faltantes, ["nome_arquivo", "caminho_imagem"]].head(10)
        raise FileNotFoundError(
            "Nao foi possivel localizar algumas imagens para o baseline, "
            "nem no dataset_binario nem nos caminhos originais. Exemplos:\n"
            f"{exemplos.to_string(index=False)}"
        )

    return df


def preparar_split_recortes(df_split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    registros = []
    ausentes = []

    for _, linha in df_split.iterrows():
        classe = str(linha["classe"])
        nome_arquivo = str(linha["nome_arquivo"])
        caminho_recorte = PASTA_DATASET_RECORTADO / classe / nome_arquivo

        registro = linha.to_dict()
        registro["caminho_imagem_recortes"] = str(
            caminho_recorte.relative_to(PASTA_PROJETO)
        )

        if caminho_recorte.exists():
            registros.append(registro)
        else:
            ausentes.append(registro)

    return pd.DataFrame(registros), pd.DataFrame(ausentes)


def validar_recortes_disponiveis(recortes_ausentes: pd.DataFrame):
    if recortes_ausentes.empty:
        return

    if not EXIGIR_RECORTES:
        return

    exemplos = recortes_ausentes[
        ["nome_arquivo", "split", "classe", "caminho_imagem_recortes"]
    ].head(10)
    raise FileNotFoundError(
        "Os recortes necessarios para o modelo recortes_resnet18 nao foram "
        "encontrados. O script 24 nao vai continuar porque a calibracao da "
        "triagem precisa das probabilidades dos dois modelos.\n\n"
        f"Total de recortes ausentes: {len(recortes_ausentes)}\n\n"
        "Exemplos:\n"
        f"{exemplos.to_string(index=False)}\n\n"
        "Reconstrua os artefatos de imagem antes de rodar este script:\n"
        "python scripts\\preparacao\\04_criar_dataset_binario.py\n"
        "python scripts\\caixas_yolo\\08_gerar_caixas_microondas.py\n"
        "python scripts\\caixas_yolo\\09_gerar_caixas_piloto_teste2.py\n"
        "python scripts\\caixas_yolo\\10_juntar_caixas_automaticas.py\n"
        "python scripts\\caixas_yolo\\12_aplicar_ajustes_manuais_caixas.py\n"
        "python scripts\\triagem\\32_gerar_predicoes_todos_splits.py"
    )


@torch.no_grad()
def obter_predicoes(
    modelo,
    df: pd.DataFrame,
    transformacao,
    dispositivo,
    coluna_caminho: str,
    coluna_modelo: str,
    coluna_prob: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "nome_arquivo",
                "split",
                "classe_real",
                "alvo",
                coluna_modelo,
                coluna_prob,
            ]
        )

    dataset = DatasetSementes(df, transformacao, coluna_caminho, coluna_modelo)
    carregador = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    modelo.eval()
    registros = []

    for lote in carregador:
        imagens = lote["imagem"].to(dispositivo)
        saidas = modelo(imagens)
        probabilidades = torch.softmax(saidas, dim=1)[:, INDICE_POSITIVO]

        for indice, prob in enumerate(probabilidades.cpu().tolist()):
            registros.append({
                "nome_arquivo": lote["nome_arquivo"][indice],
                "split": lote["split"][indice],
                "classe_real": lote["classe_real"][indice],
                "alvo": int(lote["alvo"][indice]),
                coluna_modelo: lote[coluna_modelo][indice],
                coluna_prob: float(prob),
            })

    return pd.DataFrame(registros)


def classificar_regra_3_zonas(probabilidade):
    if pd.isna(probabilidade):
        return "sem_predicao"
    if probabilidade >= LIMIAR_ALTO_RISCO:
        return "alto_risco"
    if probabilidade <= LIMIAR_BAIXO_RISCO:
        return "baixo_risco"
    return "incerto"


def classificar_regra_2_zonas(probabilidade):
    if pd.isna(probabilidade):
        return "sem_predicao"
    if probabilidade >= LIMIAR_ALTO_RISCO:
        return "alto_risco"
    return "incerto"


def consolidar_predicoes(
    df_split: pd.DataFrame,
    pred_baseline: pd.DataFrame,
    pred_recortes: pd.DataFrame,
    recortes_ausentes: pd.DataFrame,
) -> pd.DataFrame:
    base = df_split[["nome_arquivo", "split", "classe", "alvo", "caminho_imagem"]].copy()
    base = base.rename(columns={
        "classe": "classe_real",
        "caminho_imagem": "caminho_imagem_original",
    })

    colunas_merge = ["nome_arquivo", "split", "classe_real", "alvo"]
    predicoes = base.merge(pred_baseline, on=colunas_merge, how="left")
    predicoes = predicoes.merge(pred_recortes, on=colunas_merge, how="left")

    if not recortes_ausentes.empty:
        ausentes = recortes_ausentes[["nome_arquivo", "split"]].drop_duplicates()
        ausentes["recorte_ausente"] = True
        predicoes = predicoes.merge(ausentes, on=["nome_arquivo", "split"], how="left")
    else:
        predicoes["recorte_ausente"] = False

    predicoes["recorte_ausente"] = predicoes["recorte_ausente"].fillna(False)
    predicoes["prob_media_modelos"] = predicoes[
        ["prob_baseline_resnet18", "prob_recortes_resnet18"]
    ].mean(axis=1, skipna=True)
    predicoes["triagem_regra_3_zonas"] = predicoes["prob_media_modelos"].map(
        classificar_regra_3_zonas
    )
    predicoes["triagem_regra_2_zonas"] = predicoes["prob_media_modelos"].map(
        classificar_regra_2_zonas
    )

    colunas_inicio = [
        "nome_arquivo",
        "split",
        "classe_real",
        "alvo",
        "caminho_imagem_original",
        "caminho_imagem_baseline",
        "caminho_imagem_recortes",
        "recorte_ausente",
        "prob_baseline_resnet18",
        "prob_recortes_resnet18",
        "prob_media_modelos",
        "triagem_regra_3_zonas",
        "triagem_regra_2_zonas",
    ]
    colunas_inicio = [coluna for coluna in colunas_inicio if coluna in predicoes.columns]
    colunas_restantes = [coluna for coluna in predicoes.columns if coluna not in colunas_inicio]

    return predicoes[colunas_inicio + colunas_restantes]


def gerar_resumo(predicoes: pd.DataFrame) -> pd.DataFrame:
    resumo_split = (
        predicoes.groupby("split", dropna=False)
        .agg(
            total=("nome_arquivo", "count"),
            contaminadas=("alvo", "sum"),
            prob_baseline_disponivel=("prob_baseline_resnet18", "count"),
            prob_recortes_disponivel=("prob_recortes_resnet18", "count"),
            prob_media_disponivel=("prob_media_modelos", "count"),
            recortes_ausentes=("recorte_ausente", "sum"),
        )
        .reset_index()
    )
    resumo_split["nao_contaminadas"] = (
        resumo_split["total"] - resumo_split["contaminadas"]
    )
    return resumo_split[
        [
            "split",
            "total",
            "contaminadas",
            "nao_contaminadas",
            "prob_baseline_disponivel",
            "prob_recortes_disponivel",
            "prob_media_disponivel",
            "recortes_ausentes",
        ]
    ]


def main():
    print("=" * 60)
    print("GERANDO PREDICOES PARA TODOS OS SPLITS")
    print("=" * 60)

    PASTA_TRIAGEM_TABELAS.mkdir(parents=True, exist_ok=True)

    validar_arquivo(CAMINHO_SPLIT, "Arquivo de divisao treino/validacao/teste")
    validar_arquivo(CAMINHO_MODELO_BASELINE, "Modelo baseline")
    validar_arquivo(CAMINHO_MODELO_RECORTES, "Modelo de recortes")

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo usado: {dispositivo}")

    df_split = ler_csv_obrigatorio(CAMINHO_SPLIT)
    transformacao = criar_transformacao()

    print("Carregando modelos treinados...")
    modelo_baseline = carregar_modelo(CAMINHO_MODELO_BASELINE, dispositivo)
    modelo_recortes = carregar_modelo(CAMINHO_MODELO_RECORTES, dispositivo)

    print("Preparando datasets...")
    df_baseline = preparar_split_baseline(df_split)
    df_recortes, recortes_ausentes = preparar_split_recortes(df_split)
    validar_recortes_disponiveis(recortes_ausentes)

    if not recortes_ausentes.empty:
        print(
            "AVISO: alguns recortes nao foram encontrados e ficarao sem "
            "prob_recortes_resnet18."
        )
        print(f"Recortes ausentes: {len(recortes_ausentes)}")

    print("Gerando predicoes do baseline...")
    pred_baseline = obter_predicoes(
        modelo_baseline,
        df_baseline,
        transformacao,
        dispositivo,
        coluna_caminho="caminho_imagem_baseline",
        coluna_modelo="caminho_imagem_baseline",
        coluna_prob="prob_baseline_resnet18",
    )

    print("Gerando predicoes dos recortes...")
    pred_recortes = obter_predicoes(
        modelo_recortes,
        df_recortes,
        transformacao,
        dispositivo,
        coluna_caminho="caminho_imagem_recortes",
        coluna_modelo="caminho_imagem_recortes",
        coluna_prob="prob_recortes_resnet18",
    )

    predicoes = consolidar_predicoes(
        df_split,
        pred_baseline,
        pred_recortes,
        recortes_ausentes,
    )
    resumo = gerar_resumo(predicoes)

    predicoes.to_csv(
        CAMINHO_PREDICOES_TODOS_SPLITS, index=False, encoding="utf-8-sig"
    )
    resumo.to_csv(CAMINHO_RESUMO_PREDICOES, index=False, encoding="utf-8-sig")

    print()
    print("Resumo por split:")
    print(resumo.to_string(index=False))
    print()
    print("Arquivos gerados:")
    print(f"- {CAMINHO_PREDICOES_TODOS_SPLITS}")
    print(f"- {CAMINHO_RESUMO_PREDICOES}")
    print()
    print("Predicoes para todos os splits concluidas.")


if __name__ == "__main__":
    main()
