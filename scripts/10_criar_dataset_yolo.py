from pathlib import Path
import shutil

import pandas as pd


# ============================================================
# SCRIPT 10 - CRIAR DATASET YOLO
# ------------------------------------------------------------
# Objetivo:
# - Converter as caixas automaticas para formato YOLO detect
# - Reusar a divisao treino/validacao/teste do baseline
# - Criar data.yaml para Ultralytics
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_YOLO = PASTA_PROJETO / "saidas" / "yolo_dataset"

SPLIT_PARA_YOLO = {
    "treino": "train",
    "validacao": "val",
    "teste": "test",
}

CLASSES = ["nao_contaminada", "contaminada"]


def caminho_posix(caminho: Path) -> str:
    return caminho.as_posix()


def nome_label(nome_imagem: str) -> str:
    return f"{Path(nome_imagem).stem}.txt"


def normalizar_caixa(linha) -> tuple[float, float, float, float]:
    largura = float(linha["largura"])
    altura = float(linha["altura"])
    x1 = float(linha["x1"])
    y1 = float(linha["y1"])
    x2 = float(linha["x2"])
    y2 = float(linha["y2"])

    x_centro = ((x1 + x2) / 2) / largura
    y_centro = ((y1 + y2) / 2) / altura
    w = (x2 - x1) / largura
    h = (y2 - y1) / altura

    valores = [x_centro, y_centro, w, h]
    valores = [min(max(v, 0.0), 1.0) for v in valores]

    return tuple(valores)


def escrever_data_yaml():
    conteudo = "\n".join([
        f"path: {caminho_posix(PASTA_YOLO)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        "  0: nao_contaminada",
        "  1: contaminada",
        "",
    ])

    caminho_yaml = PASTA_YOLO / "data.yaml"
    caminho_yaml.write_text(conteudo, encoding="utf-8")
    return caminho_yaml


def main():
    print("=" * 60)
    print("CRIANDO DATASET YOLO")
    print("=" * 60)

    caminho_caixas = PASTA_TABELAS / "caixas_automaticas.csv"
    caminho_split = PASTA_TABELAS / "divisao_treino_validacao_teste.csv"

    if not caminho_caixas.exists():
        print("ERRO: caixas_automaticas.csv nao encontrado.")
        print("Execute primeiro:")
        print("python scripts\\08_gerar_caixas_microondas.py")
        print("python scripts\\08b_gerar_caixas_piloto_teste2.py")
        print("python scripts\\08c_juntar_caixas_automaticas.py")
        return

    if not caminho_split.exists():
        print("ERRO: divisao_treino_validacao_teste.csv nao encontrado.")
        print("Execute primeiro: python scripts\\06_treinar_baseline.py")
        return

    df_caixas = pd.read_csv(caminho_caixas)
    df_split = pd.read_csv(caminho_split)

    # Nao usar fallback no treino YOLO: fallback e caixa ampla de emergencia,
    # nao pseudo-rotulo confiavel.
    df_caixas = df_caixas[df_caixas["status_caixa"] == "ok"].copy()

    df = df_caixas.merge(
        df_split[["nome_arquivo", "split"]],
        left_on="nome_copiado",
        right_on="nome_arquivo",
        how="inner",
    )

    if len(df) == 0:
        print("ERRO: nenhuma imagem cruzou entre caixas e divisao.")
        return

    for split_yolo in SPLIT_PARA_YOLO.values():
        (PASTA_YOLO / "images" / split_yolo).mkdir(parents=True, exist_ok=True)
        (PASTA_YOLO / "labels" / split_yolo).mkdir(parents=True, exist_ok=True)

    registros = []

    for _, linha in df.iterrows():
        split_yolo = SPLIT_PARA_YOLO[str(linha["split"])]
        origem = Path(str(linha["arquivo_copiado"]))
        destino_imagem = PASTA_YOLO / "images" / split_yolo / str(linha["nome_copiado"])
        destino_label = PASTA_YOLO / "labels" / split_yolo / nome_label(str(linha["nome_copiado"]))

        shutil.copy2(origem, destino_imagem)

        x_centro, y_centro, w, h = normalizar_caixa(linha)
        linha_label = (
            f"{int(linha['classe_yolo'])} "
            f"{x_centro:.6f} {y_centro:.6f} {w:.6f} {h:.6f}\n"
        )
        destino_label.write_text(linha_label, encoding="utf-8")

        registros.append({
            "split": linha["split"],
            "split_yolo": split_yolo,
            "classe": linha["classe"],
            "classe_yolo": int(linha["classe_yolo"]),
            "imagem_yolo": str(destino_imagem),
            "label_yolo": str(destino_label),
            "status": "ok",
        })

    caminho_yaml = escrever_data_yaml()

    relatorio = pd.DataFrame(registros)
    caminho_relatorio = PASTA_TABELAS / "relatorio_dataset_yolo.csv"
    relatorio.to_csv(caminho_relatorio, index=False, encoding="utf-8-sig")

    resumo = (
        relatorio.groupby(["split_yolo", "classe"])
        .size()
        .reset_index(name="quantidade")
        .sort_values(["split_yolo", "classe"])
    )

    print()
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(resumo.to_string(index=False))
    print()
    print("Arquivos gerados:")
    print(f"- {PASTA_YOLO}")
    print(f"- {caminho_yaml}")
    print(f"- {caminho_relatorio}")


if __name__ == "__main__":
    main()
