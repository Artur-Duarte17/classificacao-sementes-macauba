from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# SCRIPT 26 - BASELINE COM METADADOS PARA CLASSIFICACAO
# ------------------------------------------------------------
# Objetivo:
# - Testar se metadados de lote/tratamento/origem explicam a contaminacao
# - Usar o mesmo split dos modelos de imagem
# - Comparar recall, especificidade e F1 contra os modelos atuais
# - Gerar uma conclusao objetiva sobre possivel vies de lote/tratamento
#
# Este script NAO usa pixels das imagens.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_TABELA_MESTRE = PASTA_TABELAS / "03_tabela_mestre"
PASTA_SPLIT = PASTA_TABELAS / "04_dataset_split"
PASTA_MODELOS_TABELAS = PASTA_TABELAS / "06_modelos"
PASTA_METADADOS = PASTA_MODELOS_TABELAS / "metadados"
PASTA_COMPARACAO = PASTA_MODELOS_TABELAS / "comparacao"

CAMINHO_TABELA_MESTRE_PADRAO = PASTA_TABELA_MESTRE / "tabela_mestre.csv"
CAMINHO_TABELA_MESTRE_ALTERNATIVO = PASTA_TABELAS / "tabela_mestre.csv"
CAMINHO_SPLIT = PASTA_SPLIT / "divisao_treino_validacao_teste.csv"

NOME_MODELO = "metadados_taxas_suavizadas"
CLASSE_POSITIVA = "contaminada"
INDICE_POSITIVO = 1
RECALL_MINIMO_PRIORITARIO = 0.95
MIN_AMOSTRAS_GRUPO = 10
ALPHA_SUAVIZACAO = 10.0
EPS = 1e-6

CAMINHO_METRICAS = PASTA_METADADOS / "metricas_metadados_teste.csv"
CAMINHO_PREDICOES = PASTA_METADADOS / "predicoes_metadados_teste.csv"
CAMINHO_THRESHOLDS = PASTA_METADADOS / "curva_threshold_metadados_validacao.csv"
CAMINHO_IMPORTANCIA = PASTA_METADADOS / "importancia_metadados_taxas.csv"
CAMINHO_TAXAS_GRUPO = PASTA_METADADOS / "taxas_contaminacao_metadados_por_grupo.csv"
CAMINHO_INDICADORES = PASTA_METADADOS / "indicadores_vies_metadados.csv"
CAMINHO_CONCLUSAO = PASTA_METADADOS / "conclusao_vies_metadados.txt"
CAMINHO_COMPARACAO = PASTA_COMPARACAO / "comparacao_metadados_vs_modelos_teste.csv"

ARQUIVOS_METRICAS_IMAGEM = [
    {
        "modelo": "baseline_resnet18_imagem_inteira",
        "caminho": PASTA_MODELOS_TABELAS / "baseline" / "metricas_baseline_resnet18_teste.csv",
    },
    {
        "modelo": "yolo_caixas_automaticas",
        "caminho": PASTA_MODELOS_TABELAS / "yolo" / "metricas_yolo_teste.csv",
    },
    {
        "modelo": "recortes_resnet18",
        "caminho": PASTA_MODELOS_TABELAS / "recortes" / "metricas_recortes_resnet18_teste.csv",
    },
]

COLUNAS_CATEGORICAS = [
    "origem",
    "origem_planilha",
    "experimento_rotulo",
    "experimento_img",
    "tratamento_planilha",
    "tratamento_normalizado",
    "pasta_esperada",
    "pasta_pai",
    "pasta_normalizada",
    "pasta_familia",
    "subpasta_caminho",
    "origem_tratamento",
    "origem_pasta",
    "experimento_tratamento",
    "experimento_pasta",
    "prefixo_id_semente",
    "primeiro_caractere_id",
    "faixa_id_semente",
    "tem_letra_id",
    "tem_numero_id",
    "extensao",
    "modo_cor",
]

COLUNAS_NUMERICAS = [
    "largura",
    "altura",
    "proporcao_imagem",
    "megapixels",
    "qtd_observacoes",
    "numero_id_semente",
    "numero_pasta",
]

COLUNAS_NUMERICAS_BINADAS = [f"{coluna}_faixa" for coluna in COLUNAS_NUMERICAS]
COLUNAS_FEATURES = COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS_BINADAS

COLUNAS_GRUPO = [
    "origem",
    "origem_planilha",
    "experimento_rotulo",
    "tratamento_planilha",
    "pasta_esperada",
    "pasta_pai",
    "origem_tratamento",
    "origem_pasta",
    "experimento_tratamento",
]

COLUNAS_COMPARACAO = [
    "tipo_entrada",
    "modelo",
    "cenario",
    "threshold",
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
    "arquivo_origem",
]


def localizar_tabela_mestre() -> Path:
    if CAMINHO_TABELA_MESTRE_PADRAO.exists():
        return CAMINHO_TABELA_MESTRE_PADRAO
    if CAMINHO_TABELA_MESTRE_ALTERNATIVO.exists():
        return CAMINHO_TABELA_MESTRE_ALTERNATIVO
    raise FileNotFoundError(
        "Tabela mestre nao encontrada em "
        f"{CAMINHO_TABELA_MESTRE_PADRAO} nem em {CAMINHO_TABELA_MESTRE_ALTERNATIVO}"
    )


def nome_seguro(caminho_relativo: str) -> str:
    texto = str(caminho_relativo)
    substituicoes = {
        "\\": "__",
        "/": "__",
        " ": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
    }
    for antigo, novo in substituicoes.items():
        texto = texto.replace(antigo, novo)
    return texto


def normalizar_texto(valor) -> str:
    if pd.isna(valor):
        return "desconhecido"
    texto = str(valor).strip().lower()
    texto = re.sub(r"\s+", "_", texto)
    texto = re.sub(r"[^a-z0-9_]+", "_", texto)
    texto = re.sub(r"_+", "_", texto).strip("_")
    return texto or "desconhecido"


def extrair_partes_id(valor) -> dict:
    texto = "" if pd.isna(valor) else str(valor).strip().lower()
    match_prefixo = re.match(r"^[a-z]+", texto)
    match_numero = re.search(r"\d+", texto)

    numero = np.nan
    if match_numero:
        numero = float(match_numero.group(0))

    if np.isnan(numero):
        faixa = "sem_numero"
    elif numero <= 5:
        faixa = "001_005"
    elif numero <= 10:
        faixa = "006_010"
    elif numero <= 20:
        faixa = "011_020"
    elif numero <= 40:
        faixa = "021_040"
    else:
        faixa = "041_mais"

    return {
        "prefixo_id_semente": match_prefixo.group(0) if match_prefixo else "sem_prefixo",
        "primeiro_caractere_id": texto[:1] if texto else "vazio",
        "numero_id_semente": numero,
        "faixa_id_semente": faixa,
        "tem_letra_id": "sim" if re.search(r"[a-z]", texto) else "nao",
        "tem_numero_id": "sim" if re.search(r"\d", texto) else "nao",
    }


def extrair_numero_pasta(valor) -> float:
    texto = "" if pd.isna(valor) else str(valor).lower()
    numeros = re.findall(r"\d+", texto)
    if not numeros:
        return np.nan
    return float(numeros[-1])


def criar_faixa_numerica(serie: pd.Series) -> pd.Series:
    valores = pd.to_numeric(serie, errors="coerce")
    if valores.notna().sum() == 0:
        return pd.Series(["sem_valor"] * len(serie), index=serie.index)

    if valores.dropna().nunique() <= 8:
        return valores.map(lambda x: "sem_valor" if pd.isna(x) else f"valor_{x:g}")

    try:
        faixas = pd.qcut(valores, q=5, duplicates="drop")
        return faixas.astype(str).replace("nan", "sem_valor")
    except ValueError:
        return valores.map(lambda x: "sem_valor" if pd.isna(x) else f"valor_{x:g}")


def carregar_base_experimento() -> pd.DataFrame:
    caminho_tabela = localizar_tabela_mestre()

    if not CAMINHO_SPLIT.exists():
        raise FileNotFoundError(f"Split nao encontrado: {CAMINHO_SPLIT}")

    tabela = pd.read_csv(caminho_tabela)
    split = pd.read_csv(CAMINHO_SPLIT)

    tabela = tabela[
        (tabela["status"] == "ok")
        & (tabela["imagem_valida"].astype(str).str.lower() == "true")
        & (tabela["classe"].isin(["contaminada", "nao_contaminada"]))
    ].copy()
    tabela["nome_copiado"] = tabela["caminho_relativo"].map(nome_seguro)

    duplicados = tabela[tabela["nome_copiado"].duplicated(keep=False)]
    if not duplicados.empty:
        caminho = PASTA_METADADOS / "duplicatas_nome_copiado_metadados.csv"
        PASTA_METADADOS.mkdir(parents=True, exist_ok=True)
        duplicados.to_csv(caminho, index=False, encoding="utf-8-sig")
        raise ValueError(f"Ha nomes copiados duplicados. Conferir: {caminho}")

    base = split.merge(
        tabela,
        left_on="nome_arquivo",
        right_on="nome_copiado",
        how="left",
        suffixes=("_split", ""),
        validate="one_to_one",
    )

    sem_metadados = base[base["nome_copiado"].isna()].copy()
    if not sem_metadados.empty:
        caminho = PASTA_METADADOS / "split_sem_metadados.csv"
        PASTA_METADADOS.mkdir(parents=True, exist_ok=True)
        sem_metadados.to_csv(caminho, index=False, encoding="utf-8-sig")
        raise ValueError(f"Ha imagens do split sem metadados. Conferir: {caminho}")

    base["classe_real"] = base["classe_split"]
    base["alvo_real"] = base["alvo"].astype(int)
    base["origem"] = base["origem_planilha"]

    caminho_partes = (
        base["caminho_relativo"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.split("/", expand=True)
    )
    base["experimento_caminho"] = caminho_partes[0].fillna("desconhecido")
    base["subpasta_caminho"] = caminho_partes[1].fillna("desconhecido")

    base["tratamento_normalizado"] = base["tratamento_planilha"].map(normalizar_texto)
    base["pasta_normalizada"] = base["pasta_esperada"].map(normalizar_texto)
    base["pasta_familia"] = (
        base["pasta_esperada"]
        .map(normalizar_texto)
        .str.replace(r"_?\d+$", "", regex=True)
        .replace("", "desconhecido")
    )

    base["origem_tratamento"] = (
        base["origem"].map(normalizar_texto) + "__" + base["tratamento_planilha"].map(normalizar_texto)
    )
    base["origem_pasta"] = (
        base["origem"].map(normalizar_texto) + "__" + base["pasta_esperada"].map(normalizar_texto)
    )
    base["experimento_tratamento"] = (
        base["experimento_rotulo"].map(normalizar_texto)
        + "__"
        + base["tratamento_planilha"].map(normalizar_texto)
    )
    base["experimento_pasta"] = (
        base["experimento_rotulo"].map(normalizar_texto)
        + "__"
        + base["pasta_esperada"].map(normalizar_texto)
    )

    partes_id = base["id_busca"].apply(extrair_partes_id).apply(pd.Series)
    base = pd.concat([base, partes_id], axis=1)
    base["numero_pasta"] = base["pasta_esperada"].apply(extrair_numero_pasta)

    base["largura"] = pd.to_numeric(base["largura"], errors="coerce")
    base["altura"] = pd.to_numeric(base["altura"], errors="coerce")
    base["qtd_observacoes"] = pd.to_numeric(base["qtd_observacoes"], errors="coerce")
    base["proporcao_imagem"] = base["largura"] / base["altura"].replace(0, np.nan)
    base["megapixels"] = (base["largura"] * base["altura"]) / 1_000_000

    for coluna in COLUNAS_CATEGORICAS:
        if coluna not in base.columns:
            base[coluna] = "desconhecido"
        base[coluna] = base[coluna].fillna("desconhecido").astype(str)

    for coluna in COLUNAS_NUMERICAS:
        if coluna not in base.columns:
            base[coluna] = np.nan
        base[coluna] = pd.to_numeric(base[coluna], errors="coerce")

    for coluna in COLUNAS_NUMERICAS:
        base[f"{coluna}_faixa"] = (
            criar_faixa_numerica(base[coluna])
            .fillna("sem_valor")
            .astype(str)
        )

    return base


def logit(probabilidade) -> np.ndarray:
    p = np.clip(probabilidade, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(valor) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(valor)))


def treinar_modelo_metadados(treino: pd.DataFrame) -> dict:
    taxa_global = float(treino["alvo_real"].mean())
    estatisticas = []
    importancia = []

    for coluna in COLUNAS_FEATURES:
        resumo = (
            treino.groupby(coluna, dropna=False)["alvo_real"]
            .agg(total="size", contaminadas="sum")
            .reset_index()
        )
        resumo["taxa_observada"] = resumo["contaminadas"] / resumo["total"].replace(0, np.nan)
        resumo["taxa_suavizada"] = (
            resumo["contaminadas"] + ALPHA_SUAVIZACAO * taxa_global
        ) / (resumo["total"] + ALPHA_SUAVIZACAO)

        grupos_validos = resumo[resumo["total"] >= MIN_AMOSTRAS_GRUPO].copy()
        if len(grupos_validos) >= 2:
            taxa_minima = float(grupos_validos["taxa_observada"].min())
            taxa_maxima = float(grupos_validos["taxa_observada"].max())
        else:
            taxa_minima = float(resumo["taxa_observada"].min())
            taxa_maxima = float(resumo["taxa_observada"].max())

        amplitude = taxa_maxima - taxa_minima
        peso = max(float(amplitude), 0.01)
        mapa = dict(zip(resumo[coluna].astype(str), resumo["taxa_suavizada"]))

        estatisticas.append({
            "campo": coluna,
            "mapa_taxa": mapa,
            "peso": peso,
            "taxa_global": taxa_global,
        })
        importancia.append({
            "campo": coluna,
            "peso_modelo": peso,
            "amplitude_taxa_treino": amplitude,
            "taxa_minima_treino": taxa_minima,
            "taxa_maxima_treino": taxa_maxima,
            "categorias_treino": int(len(resumo)),
            "categorias_com_min_amostras": int(len(grupos_validos)),
        })

    return {
        "taxa_global": taxa_global,
        "estatisticas": estatisticas,
        "importancia": pd.DataFrame(importancia).sort_values(
            ["peso_modelo", "amplitude_taxa_treino"],
            ascending=[False, False],
        ),
    }


def predizer_modelo_metadados(modelo: dict, df: pd.DataFrame) -> np.ndarray:
    soma_logits = np.zeros(len(df), dtype=float)
    soma_pesos = 0.0

    for estatistica in modelo["estatisticas"]:
        coluna = estatistica["campo"]
        peso = float(estatistica["peso"])
        taxas = (
            df[coluna]
            .astype(str)
            .map(estatistica["mapa_taxa"])
            .fillna(estatistica["taxa_global"])
            .astype(float)
            .to_numpy()
        )
        soma_logits += peso * logit(taxas)
        soma_pesos += peso

    if soma_pesos <= 0:
        return np.full(len(df), modelo["taxa_global"], dtype=float)

    return sigmoid(soma_logits / soma_pesos)


def calcular_metricas(y_real, prob_contaminada, threshold: float) -> dict:
    y_real = np.asarray(y_real).astype(int)
    pred = (prob_contaminada >= threshold).astype(int)

    tn = int(((y_real == 0) & (pred == 0)).sum())
    fp = int(((y_real == 0) & (pred == 1)).sum())
    fn = int(((y_real == 1) & (pred == 0)).sum())
    tp = int(((y_real == 1) & (pred == 1)).sum())

    precisao = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precisao * recall / max(precisao + recall, EPS)
    sensibilidade = tp / max(tp + fn, 1)
    especificidade = tn / max(tn + fp, 1)

    return {
        "modelo": NOME_MODELO,
        "threshold": float(threshold),
        "acuracia": float((y_real == pred).mean()),
        "precisao_contaminada": float(precisao),
        "recall_contaminada": float(recall),
        "sensibilidade_contaminada": float(sensibilidade),
        "especificidade_nao_contaminada": float(especificidade),
        "f1_contaminada": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def gerar_curva_threshold(df_validacao: pd.DataFrame) -> pd.DataFrame:
    y_real = df_validacao["alvo_real"].to_numpy()
    prob = df_validacao["prob_contaminada_metadados"].to_numpy()
    registros = []

    for threshold in np.arange(0.01, 1.00, 0.01):
        registros.append(calcular_metricas(y_real, prob, round(float(threshold), 2)))

    return pd.DataFrame(registros)


def escolher_threshold_por_f1(df_thresholds: pd.DataFrame) -> float:
    melhor = df_thresholds.sort_values(
        ["f1_contaminada", "recall_contaminada", "especificidade_nao_contaminada"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(melhor["threshold"])


def escolher_threshold_por_recall(df_thresholds: pd.DataFrame) -> float:
    candidatos = df_thresholds[
        df_thresholds["recall_contaminada"] >= RECALL_MINIMO_PRIORITARIO
    ].copy()

    if candidatos.empty:
        candidatos = df_thresholds.copy()
        ordenacao = ["recall_contaminada", "f1_contaminada", "precisao_contaminada"]
    else:
        ordenacao = ["f1_contaminada", "recall_contaminada", "precisao_contaminada"]

    melhor = candidatos.sort_values(ordenacao, ascending=[False, False, False]).iloc[0]
    return float(melhor["threshold"])


def adicionar_predicoes_threshold(df: pd.DataFrame, sufixo: str, threshold: float) -> pd.DataFrame:
    coluna = f"predito_threshold_{sufixo}"
    df[coluna] = np.where(
        df["prob_contaminada_metadados"] >= threshold,
        "contaminada",
        "nao_contaminada",
    )
    return df


def calcular_taxas_por_grupo(base: pd.DataFrame) -> pd.DataFrame:
    registros = []

    for split_nome, df_split in [("todos", base), *list(base.groupby("split"))]:
        for coluna in COLUNAS_GRUPO:
            resumo = (
                df_split.groupby(coluna, dropna=False)["alvo_real"]
                .agg(total="size", contaminadas="sum")
                .reset_index()
                .rename(columns={coluna: "valor"})
            )
            resumo["campo"] = coluna
            resumo["split"] = split_nome
            resumo["nao_contaminadas"] = resumo["total"] - resumo["contaminadas"]
            resumo["taxa_contaminacao"] = resumo["contaminadas"] / resumo["total"].replace(0, np.nan)
            registros.append(resumo)

    return pd.concat(registros, ignore_index=True).sort_values(
        ["campo", "split", "taxa_contaminacao", "total"],
        ascending=[True, True, False, False],
    )


def carregar_metricas_modelos_imagem() -> pd.DataFrame:
    tabelas = []
    for item in ARQUIVOS_METRICAS_IMAGEM:
        caminho = item["caminho"]
        if not caminho.exists():
            print(f"AVISO: metricas ausentes: {caminho}")
            continue

        df = pd.read_csv(caminho)
        if "modelo" not in df.columns:
            df.insert(0, "modelo", item["modelo"])
        else:
            df["modelo"] = df["modelo"].fillna(item["modelo"])

        if "sensibilidade_contaminada" not in df.columns and "recall_contaminada" in df.columns:
            df["sensibilidade_contaminada"] = df["recall_contaminada"]
        if "especificidade_nao_contaminada" not in df.columns:
            tn = pd.to_numeric(df.get("tn", 0), errors="coerce").fillna(0)
            fp = pd.to_numeric(df.get("fp", 0), errors="coerce").fillna(0)
            df["especificidade_nao_contaminada"] = tn / (tn + fp).replace(0, np.nan)

        df["tipo_entrada"] = "imagem"
        df["arquivo_origem"] = str(caminho.relative_to(PASTA_PROJETO))
        tabelas.append(df)

    if not tabelas:
        return pd.DataFrame()

    return pd.concat(tabelas, ignore_index=True)


def criar_comparacao(metricas_metadados: pd.DataFrame) -> pd.DataFrame:
    metricas_imagem = carregar_metricas_modelos_imagem()
    metadados = metricas_metadados.copy()
    metadados["tipo_entrada"] = "metadados"
    metadados["arquivo_origem"] = str(CAMINHO_METRICAS.relative_to(PASTA_PROJETO))

    comparacao = pd.concat([metricas_imagem, metadados], ignore_index=True, sort=False)
    for coluna in COLUNAS_COMPARACAO:
        if coluna not in comparacao.columns:
            comparacao[coluna] = np.nan

    return comparacao[COLUNAS_COMPARACAO].sort_values(
        ["recall_contaminada", "f1_contaminada", "especificidade_nao_contaminada"],
        ascending=[False, False, False],
    )


def avaliar_indicio_vies(
    metricas_metadados: pd.DataFrame,
    comparacao: pd.DataFrame,
    taxas_grupo: pd.DataFrame,
    importancia: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    linha_metadados = metricas_metadados[
        metricas_metadados["cenario"] == "teste_threshold_melhor_f1_validacao"
    ].iloc[0]

    imagem = comparacao[comparacao["tipo_entrada"] == "imagem"].copy()
    if imagem.empty:
        melhor_imagem = pd.Series({
            "f1_contaminada": np.nan,
            "recall_contaminada": np.nan,
            "especificidade_nao_contaminada": np.nan,
        })
    else:
        melhor_imagem = imagem.sort_values(
            ["f1_contaminada", "recall_contaminada", "especificidade_nao_contaminada"],
            ascending=[False, False, False],
        ).iloc[0]

    taxas_todos = taxas_grupo[
        (taxas_grupo["split"] == "todos") & (taxas_grupo["total"] >= MIN_AMOSTRAS_GRUPO)
    ].copy()
    amplitudes = (
        taxas_todos.groupby("campo")["taxa_contaminacao"]
        .agg(taxa_minima="min", taxa_maxima="max")
        .reset_index()
    )
    amplitudes["amplitude_taxa_contaminacao"] = (
        amplitudes["taxa_maxima"] - amplitudes["taxa_minima"]
    )
    maior_amplitude = amplitudes.sort_values(
        "amplitude_taxa_contaminacao", ascending=False
    ).head(1)

    if maior_amplitude.empty:
        campo_maior_amplitude = "nenhum"
        amplitude_maxima = 0.0
    else:
        campo_maior_amplitude = str(maior_amplitude.iloc[0]["campo"])
        amplitude_maxima = float(maior_amplitude.iloc[0]["amplitude_taxa_contaminacao"])

    top_importancia = importancia.head(20)
    termos_lote = ["origem", "tratamento", "pasta", "experimento"]
    qtd_top_lote = int(
        top_importancia["campo"]
        .str.contains("|".join(termos_lote), case=False, regex=True)
        .sum()
    )

    f1_meta = float(linha_metadados["f1_contaminada"])
    recall_meta = float(linha_metadados["recall_contaminada"])
    espec_meta = float(linha_metadados["especificidade_nao_contaminada"])
    f1_img = float(melhor_imagem["f1_contaminada"]) if pd.notna(melhor_imagem["f1_contaminada"]) else np.nan
    recall_img = (
        float(melhor_imagem["recall_contaminada"])
        if pd.notna(melhor_imagem["recall_contaminada"])
        else np.nan
    )

    competitividade_f1 = f1_meta / f1_img if f1_img and not np.isnan(f1_img) else np.nan
    competitividade_recall = (
        recall_meta / recall_img if recall_img and not np.isnan(recall_img) else np.nan
    )

    desempenho_competitivo = (
        pd.notna(competitividade_f1)
        and competitividade_f1 >= 0.90
        and pd.notna(competitividade_recall)
        and competitividade_recall >= 0.90
    )
    separacao_grupos_forte = amplitude_maxima >= 0.40
    importancia_concentrada_em_lote = qtd_top_lote >= 8

    if desempenho_competitivo and separacao_grupos_forte:
        nivel = "forte"
        conclusao = (
            "Ha indicio forte de vies de lote/tratamento: o modelo sem pixels chegou "
            "perto dos modelos de imagem e ha grande diferenca de taxa de contaminacao "
            f"entre grupos de {campo_maior_amplitude}."
        )
    elif desempenho_competitivo or (separacao_grupos_forte and importancia_concentrada_em_lote):
        nivel = "moderado"
        conclusao = (
            "Ha indicio moderado de vies de lote/tratamento: parte relevante da predicao "
            "parece explicavel por metadados, mas o sinal nao fecha todos os criterios fortes."
        )
    else:
        nivel = "fraco"
        conclusao = (
            "Nao ha indicio forte de vies de lote/tratamento pelo criterio deste script: "
            "os metadados nao ficaram competitivos o suficiente ou a separacao por grupos foi limitada."
        )

    indicadores = pd.DataFrame([
        {
            "indicador": "nivel_indicio_vies_lote_tratamento",
            "valor": nivel,
            "interpretacao": conclusao,
        },
        {
            "indicador": "f1_metadados_melhor_f1_validacao",
            "valor": f1_meta,
            "interpretacao": "F1 no teste usando apenas metadados e threshold calibrado na validacao.",
        },
        {
            "indicador": "recall_metadados_melhor_f1_validacao",
            "valor": recall_meta,
            "interpretacao": "Recall da classe contaminada no teste usando apenas metadados.",
        },
        {
            "indicador": "especificidade_metadados_melhor_f1_validacao",
            "valor": espec_meta,
            "interpretacao": "Especificidade da classe nao contaminada no teste usando apenas metadados.",
        },
        {
            "indicador": "f1_melhor_modelo_imagem",
            "valor": f1_img,
            "interpretacao": "Melhor F1 observado entre os modelos de imagem carregados.",
        },
        {
            "indicador": "competitividade_f1_metadados_vs_imagem",
            "valor": competitividade_f1,
            "interpretacao": "Razao entre F1 do baseline de metadados e melhor F1 de imagem.",
        },
        {
            "indicador": "competitividade_recall_metadados_vs_imagem",
            "valor": competitividade_recall,
            "interpretacao": "Razao entre recall do baseline de metadados e melhor recall de imagem.",
        },
        {
            "indicador": "maior_amplitude_taxa_contaminacao_grupo",
            "valor": amplitude_maxima,
            "interpretacao": (
                "Maior diferenca max-min na taxa de contaminacao entre grupos "
                f"com pelo menos {MIN_AMOSTRAS_GRUPO} amostras."
            ),
        },
        {
            "indicador": "campo_maior_amplitude_taxa_contaminacao",
            "valor": campo_maior_amplitude,
            "interpretacao": "Campo de metadados com maior separacao bruta entre grupos.",
        },
        {
            "indicador": "top20_importancias_de_lote_tratamento",
            "valor": qtd_top_lote,
            "interpretacao": "Quantidade de features de origem/tratamento/pasta/experimento no top 20 do modelo tabular.",
        },
    ])

    texto = "\n".join([
        "Conclusao do baseline de metadados",
        "=" * 42,
        conclusao,
        "",
        f"Nivel: {nivel}",
        f"F1 metadados: {f1_meta:.3f}",
        f"Recall metadados: {recall_meta:.3f}",
        f"Especificidade metadados: {espec_meta:.3f}",
        f"F1 melhor imagem: {f1_img:.3f}" if not np.isnan(f1_img) else "F1 melhor imagem: nao disponivel",
        (
            f"Competitividade F1 metadados/imagem: {competitividade_f1:.3f}"
            if not np.isnan(competitividade_f1)
            else "Competitividade F1 metadados/imagem: nao disponivel"
        ),
        f"Maior amplitude de taxa por grupo: {amplitude_maxima:.3f} ({campo_maior_amplitude})",
        f"Features de lote/tratamento no top 20: {qtd_top_lote}",
        "",
        "Criterio usado:",
        "- Forte: F1 e recall dos metadados >= 90% do melhor modelo de imagem, "
        "e amplitude de contaminacao por grupo >= 0.40.",
        "- Moderado: apenas parte desses sinais aparece de forma consistente.",
        "- Fraco: os sinais acima nao sao suficientes.",
    ])

    return indicadores, texto


def main():
    print("=" * 60)
    print("BASELINE COM METADADOS PARA CLASSIFICACAO")
    print("=" * 60)

    PASTA_METADADOS.mkdir(parents=True, exist_ok=True)
    PASTA_COMPARACAO.mkdir(parents=True, exist_ok=True)

    base = carregar_base_experimento()
    print(f"Registros com split e metadados: {len(base)}")
    print(pd.crosstab(base["split"], base["classe_real"]).to_string())

    treino = base[base["split"] == "treino"].copy()
    validacao = base[base["split"] == "validacao"].copy()
    teste = base[base["split"] == "teste"].copy()

    modelo = treinar_modelo_metadados(treino)

    for nome_split, df_split in [("treino", treino), ("validacao", validacao), ("teste", teste)]:
        prob = predizer_modelo_metadados(modelo, df_split)
        base.loc[df_split.index, "prob_contaminada_metadados"] = prob
        print(f"Probabilidades geradas para {nome_split}: {len(df_split)}")

    validacao = base[base["split"] == "validacao"].copy()
    teste = base[base["split"] == "teste"].copy()

    thresholds = gerar_curva_threshold(validacao)
    thresholds.to_csv(CAMINHO_THRESHOLDS, index=False, encoding="utf-8-sig")

    threshold_f1 = escolher_threshold_por_f1(thresholds)
    threshold_recall = escolher_threshold_por_recall(thresholds)

    y_teste = teste["alvo_real"].to_numpy()
    prob_teste = teste["prob_contaminada_metadados"].to_numpy()

    metricas = pd.DataFrame([
        {
            "cenario": "teste_threshold_0_50",
            **calcular_metricas(y_teste, prob_teste, threshold=0.50),
        },
        {
            "cenario": "teste_threshold_melhor_f1_validacao",
            **calcular_metricas(y_teste, prob_teste, threshold=threshold_f1),
        },
        {
            "cenario": "teste_threshold_prioridade_recall_validacao",
            **calcular_metricas(y_teste, prob_teste, threshold=threshold_recall),
        },
    ])
    metricas.to_csv(CAMINHO_METRICAS, index=False, encoding="utf-8-sig")

    predicoes = teste[[
        "caminho_imagem",
        "nome_arquivo_split",
        "classe_real",
        "alvo_real",
        "split",
        "prob_contaminada_metadados",
        *COLUNAS_CATEGORICAS,
        *COLUNAS_NUMERICAS,
        *COLUNAS_NUMERICAS_BINADAS,
    ]].copy()
    predicoes = adicionar_predicoes_threshold(predicoes, "0_50", 0.50)
    predicoes = adicionar_predicoes_threshold(predicoes, "melhor_f1_validacao", threshold_f1)
    predicoes = adicionar_predicoes_threshold(predicoes, "prioridade_recall_validacao", threshold_recall)
    predicoes.to_csv(CAMINHO_PREDICOES, index=False, encoding="utf-8-sig")

    importancia = modelo["importancia"]
    importancia.to_csv(CAMINHO_IMPORTANCIA, index=False, encoding="utf-8-sig")

    taxas_grupo = calcular_taxas_por_grupo(base)
    taxas_grupo.to_csv(CAMINHO_TAXAS_GRUPO, index=False, encoding="utf-8-sig")

    comparacao = criar_comparacao(metricas)
    comparacao.to_csv(CAMINHO_COMPARACAO, index=False, encoding="utf-8-sig")

    indicadores, texto_conclusao = avaliar_indicio_vies(
        metricas,
        comparacao,
        taxas_grupo,
        importancia,
    )
    indicadores.to_csv(CAMINHO_INDICADORES, index=False, encoding="utf-8-sig")
    CAMINHO_CONCLUSAO.write_text(texto_conclusao, encoding="utf-8")

    print()
    print("Metricas do baseline de metadados no teste:")
    print(metricas.to_string(index=False))

    print()
    print("Comparacao com modelos atuais:")
    print(comparacao.to_string(index=False))

    print()
    print(texto_conclusao)

    print()
    print("Arquivos gerados:")
    for caminho in [
        CAMINHO_METRICAS,
        CAMINHO_PREDICOES,
        CAMINHO_THRESHOLDS,
        CAMINHO_IMPORTANCIA,
        CAMINHO_TAXAS_GRUPO,
        CAMINHO_INDICADORES,
        CAMINHO_CONCLUSAO,
        CAMINHO_COMPARACAO,
    ]:
        print(f"- {caminho}")


if __name__ == "__main__":
    main()
