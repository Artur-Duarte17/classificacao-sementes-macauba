from __future__ import annotations

from pathlib import Path


PASTA_PROJETO = Path(__file__).resolve().parents[3]
PASTA_DATASET_RECORTADO = PASTA_PROJETO / "saidas" / "dataset_recortado"
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_TABELA_MESTRE = PASTA_TABELAS / "03_tabela_mestre"
PASTA_CLASSICOS = PASTA_TABELAS / "06_modelos" / "classicos"
PASTA_VALIDACAO = PASTA_TABELAS / "07_classificacao_final" / "validacao_tratamento"
PASTA_HISTORICOS_MOBILENET = PASTA_VALIDACAO / "historicos_mobilenetv2"
PASTA_CHECKPOINTS = PASTA_PROJETO / "saidas" / "modelos" / "validacao_tratamento"

CAMINHO_TABELA_MESTRE_PADRAO = PASTA_TABELA_MESTRE / "tabela_mestre.csv"
CAMINHO_TABELA_MESTRE_ALTERNATIVO = PASTA_TABELAS / "tabela_mestre.csv"
CAMINHO_ATRIBUTOS = PASTA_CLASSICOS / "atributos_visuais_recortes.csv"
CAMINHO_FEATURES = PASTA_CLASSICOS / "features_classicos.csv"
CAMINHO_COMPARACAO_SPLIT_ORIGINAL = (
    PASTA_TABELAS / "07_classificacao_final" / "comparacao_final_classificacao.csv"
)

CAMINHO_FOLDS = PASTA_VALIDACAO / "folds_validacao_por_tratamento.csv"
CAMINHO_PREDICOES = PASTA_VALIDACAO / "predicoes_validacao_por_tratamento.csv"
CAMINHO_METRICAS = PASTA_VALIDACAO / "metricas_validacao_por_tratamento.csv"
CAMINHO_THRESHOLDS = PASTA_VALIDACAO / "thresholds_validacao_por_tratamento.csv"
CAMINHO_RESUMO = PASTA_VALIDACAO / "resumo_generalizacao_por_tratamento.csv"
CAMINHO_COMPARACAO_PROTOCOLOS = (
    PASTA_VALIDACAO / "comparacao_split_original_vs_tratamento.csv"
)
CAMINHO_CONFIG = PASTA_VALIDACAO / "config_validacao_por_tratamento.json"
CAMINHO_DIAGNOSTICO_FOLDS = (
    PASTA_VALIDACAO / "diagnostico_folds_validacao_por_tratamento.csv"
)
CAMINHO_DIAGNOSTICO_CV = (
    PASTA_VALIDACAO / "diagnostico_cv_interna_validacao_por_tratamento.csv"
)

CLASSES = ["nao_contaminada", "contaminada"]
CLASSE_POSITIVA = "contaminada"
CLASSE_PARA_INDICE = {classe: indice for indice, classe in enumerate(CLASSES)}
INDICE_POSITIVO = CLASSE_PARA_INDICE[CLASSE_POSITIVA]
GRUPO_VALIDACAO_EXTERNA = "experimento_tratamento"
SEMENTE_ALEATORIA = 42
RECALL_MINIMO_PRIORITARIO = 0.95
EPS = 1e-12

CONJUNTO_PRINCIPAL = "principal_normalizado"
CONJUNTO_NAO_APLICAVEL = "nao_aplicavel"
PROTOCOLO = "leave_one_experimento_tratamento_out"

CV_FOLDS_MAX = 5
N_JOBS_GRID = 6

TAMANHO_IMAGEM = 224
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
USAR_MIXED_PRECISION = True
USAR_CHANNELS_LAST_CUDA = True
EPOCHS_TOTAL = 80
EPOCHS_BACKBONE_CONGELADO = 5
PACIENCIA_EARLY_STOPPING = 8
LEARNING_RATE_CLASSIFICADOR = 1e-4
LEARNING_RATE_AJUSTE_FINO = 1e-5
WEIGHT_DECAY = 1e-4
BLOCOS_FINAIS_DESCONGELADOS = 4
PESOS_PRE_TREINADOS = "MobileNet_V2_Weights.DEFAULT"
PESOS_IMAGENET_CARREGADOS = True

MIN_AMOSTRAS_GRUPO_VALIDACAO = 10
MIN_AMOSTRAS_GRUPO_METADADOS = 10
ALPHA_SUAVIZACAO = 10.0

MODELOS_TREINAVEIS = ["random_forest", "svm_rbf", "metadados", "mobilenetv2"]
TODOS_MODELOS = [*MODELOS_TREINAVEIS, "baseline_sempre_contaminada"]

COLUNAS_EXCLUIDAS_OBRIGATORIAS = {
    "nome_arquivo",
    "split",
    "classe",
    "classe_real",
    "alvo",
    "caminho_imagem",
    "caminho_recorte",
    "status_atributos",
    "erro_atributos",
}

COLUNAS_ABSOLUTAS_EXCLUIR_PRINCIPAL = {
    "largura_recorte",
    "altura_recorte",
    "pixels_recorte",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "mask_area_px",
    "contour_area_px",
    "contour_perimeter_px",
}

PREFIXOS_TEXTURA_RESOLUCAO_ORIGINAL = ("texture_", "lbp_")
TERMOS_METADADOS_PROIBIDOS_FEATURES = (
    "origem",
    "tratamento",
    "pasta",
    "experimento",
    "caminho",
    "arquivo",
    "nome",
    "split",
    "classe",
    "alvo",
    "status",
    "erro",
)

COLUNAS_CATEGORICAS_METADADOS = [
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

COLUNAS_NUMERICAS_METADADOS = [
    "largura",
    "altura",
    "proporcao_imagem",
    "megapixels",
    "qtd_observacoes",
    "numero_id_semente",
    "numero_pasta",
]

COLUNAS_NUMERICAS_BINADAS_METADADOS = [
    f"{coluna}_faixa" for coluna in COLUNAS_NUMERICAS_METADADOS
]
COLUNAS_FEATURES_METADADOS = (
    COLUNAS_CATEGORICAS_METADADOS + COLUNAS_NUMERICAS_BINADAS_METADADOS
)

COLUNAS_METRICAS = [
    "acuracia",
    "precisao_contaminada",
    "recall_contaminada",
    "sensibilidade_contaminada",
    "especificidade_nao_contaminada",
    "f1_contaminada",
    "balanced_accuracy",
    "youden_j",
    "mcc",
    "taxa_predita_contaminada",
]

COLUNAS_COMPARACAO_PROTOCOLOS = [
    "f1_contaminada",
    "balanced_accuracy",
    "mcc",
    "recall_contaminada",
    "especificidade_nao_contaminada",
]

CHAVES_RESULTADO = ["modelo", "grupo_externo", "cenario", "conjunto_features"]
CHAVES_PREDICAO = [
    "modelo",
    "grupo_externo",
    "cenario",
    "conjunto_features",
    "nome_arquivo",
]
CHAVES_THRESHOLD = [
    "modelo",
    "grupo_externo",
    "conjunto_features",
    "threshold",
]


def contexto_modelo(nome_modelo: str) -> dict:
    if nome_modelo == "random_forest":
        return {
            "modelo": "random_forest",
            "familia_modelo": "random_forest",
            "tipo_entrada": "atributos_visuais_recortes",
            "usa_pixels": False,
            "usa_recorte": True,
            "usa_atributos_visuais": True,
            "usa_metadados": False,
            "conjunto_features": CONJUNTO_PRINCIPAL,
            "resultado_oficial": True,
            "papel_experimento": "modelo_visual_classico",
        }
    if nome_modelo == "svm_rbf":
        return {
            "modelo": "svm_rbf",
            "familia_modelo": "svm_rbf",
            "tipo_entrada": "atributos_visuais_recortes",
            "usa_pixels": False,
            "usa_recorte": True,
            "usa_atributos_visuais": True,
            "usa_metadados": False,
            "conjunto_features": CONJUNTO_PRINCIPAL,
            "resultado_oficial": True,
            "papel_experimento": "diagnostico_comparativo",
        }
    if nome_modelo == "metadados":
        return {
            "modelo": "metadados_taxas_suavizadas",
            "familia_modelo": "baseline_metadados",
            "tipo_entrada": "metadados",
            "usa_pixels": False,
            "usa_recorte": False,
            "usa_atributos_visuais": False,
            "usa_metadados": True,
            "conjunto_features": CONJUNTO_NAO_APLICAVEL,
            "resultado_oficial": False,
            "papel_experimento": "diagnostico_vies",
        }
    if nome_modelo == "mobilenetv2":
        return {
            "modelo": "mobilenetv2_recortes",
            "familia_modelo": "cnn_mobilenetv2",
            "tipo_entrada": "recorte",
            "usa_pixels": True,
            "usa_recorte": True,
            "usa_atributos_visuais": False,
            "usa_metadados": False,
            "conjunto_features": CONJUNTO_NAO_APLICAVEL,
            "resultado_oficial": True,
            "papel_experimento": "modelo_visual",
        }
    if nome_modelo == "baseline_sempre_contaminada":
        return {
            "modelo": "baseline_sempre_contaminada",
            "familia_modelo": "controle",
            "tipo_entrada": "controle",
            "usa_pixels": False,
            "usa_recorte": False,
            "usa_atributos_visuais": False,
            "usa_metadados": False,
            "conjunto_features": CONJUNTO_NAO_APLICAVEL,
            "resultado_oficial": False,
            "papel_experimento": "controle",
        }
    raise ValueError(f"Contexto desconhecido: {nome_modelo}")


def completar_contexto_fold(contexto: dict, fold: dict) -> dict:
    return {
        **contexto,
        "protocolo": PROTOCOLO,
        "fold": int(fold["fold"]),
        "grupo_externo": fold["grupo_externo"],
        "grupo_validacao": fold["grupo_validacao"],
        "seed": SEMENTE_ALEATORIA,
    }
