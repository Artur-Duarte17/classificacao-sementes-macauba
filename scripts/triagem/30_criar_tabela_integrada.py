from pathlib import Path
import json
import re

import pandas as pd


# ============================================================
# SCRIPT 30 - CRIAR TABELA INTEGRADA DA TRIAGEM
# ------------------------------------------------------------
# Objetivo:
# - Integrar rotulos, folds externos e probabilidades do protocolo
#   leave-one-experimento-tratamento-out
# - Usar apenas predicoes externas de modelos visuais oficiais
# - Nao calibrar thresholds e nao escolher estrategias
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[2]
PASTA_TABELAS = PASTA_PROJETO / "saidas" / "tabelas"
PASTA_VALIDACAO = PASTA_TABELAS / "07_classificacao_final" / "validacao_tratamento"
PASTA_TRIAGEM = PASTA_TABELAS / "08_triagem"

CAMINHO_FOLDS = PASTA_VALIDACAO / "folds_validacao_por_tratamento.csv"
CAMINHO_PREDICOES_VALIDACAO = PASTA_VALIDACAO / "predicoes_validacao_por_tratamento.csv"
CAMINHO_THRESHOLDS_VALIDACAO = PASTA_VALIDACAO / "thresholds_validacao_por_tratamento.csv"

CAMINHO_TABELA_INTEGRADA = PASTA_TRIAGEM / "tabela_integrada_triagem.csv"
CAMINHO_THRESHOLDS_INTERNOS = PASTA_TRIAGEM / "thresholds_internos_modelos_triagem.csv"
CAMINHO_MANIFESTO = PASTA_TRIAGEM / "manifesto_integracao_triagem.json"

CENARIO_PROBABILIDADE = "teste_threshold_0_50"
MODELOS_EXCLUIDOS = {"baseline_sempre_contaminada", "metadados_taxas_suavizadas"}


def ler_csv_obrigatorio(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo obrigatorio ausente: {caminho}")
    df = pd.read_csv(caminho)
    if df.empty:
        raise ValueError(f"Arquivo obrigatorio vazio: {caminho}")
    return df


def gravar_csv_atomico(df: pd.DataFrame, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(f"{caminho.name}.tmp")
    df.to_csv(temporario, index=False, encoding="utf-8-sig")
    pd.read_csv(temporario)
    temporario.replace(caminho)


def gravar_json_atomico(objeto: dict, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(f"{caminho.name}.tmp")
    temporario.write_text(
        json.dumps(objeto, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    json.loads(temporario.read_text(encoding="utf-8"))
    temporario.replace(caminho)


def caminho_relativo(caminho: Path) -> str:
    return str(caminho.relative_to(PASTA_PROJETO))


def slug_modelo(modelo: str, conjunto_features: str) -> str:
    texto = f"{modelo}_{conjunto_features}"
    texto = re.sub(r"[^a-zA-Z0-9]+", "_", texto).strip("_").lower()
    return texto


def normalizar_bool(serie: pd.Series) -> pd.Series:
    if serie.dtype == bool:
        return serie
    return serie.astype(str).str.lower().isin(["true", "1", "sim"])


def selecionar_predicoes_visuais(predicoes: pd.DataFrame, total_amostras: int) -> tuple[pd.DataFrame, list[dict]]:
    trabalho = predicoes.copy()
    trabalho = trabalho[
        (trabalho["papel_amostra"].astype(str) == "teste_externo")
        & (trabalho["cenario"].astype(str) == CENARIO_PROBABILIDADE)
        & (~trabalho["modelo"].astype(str).isin(MODELOS_EXCLUIDOS))
        & (~normalizar_bool(trabalho.get("usa_metadados", pd.Series(False, index=trabalho.index))))
    ].copy()

    if "resultado_oficial" in trabalho.columns:
        trabalho = trabalho[normalizar_bool(trabalho["resultado_oficial"])].copy()

    if trabalho.empty:
        raise ValueError("Nenhuma predicao visual oficial encontrada para triagem.")

    modelos = []
    partes = []
    for (modelo, conjunto), grupo in trabalho.groupby(["modelo", "conjunto_features"]):
        grupo = grupo.copy()
        grupo = grupo.drop_duplicates(["modelo", "conjunto_features", "nome_arquivo"])
        if len(grupo) != total_amostras:
            continue
        slug = slug_modelo(str(modelo), str(conjunto))
        modelos.append({
            "modelo": str(modelo),
            "conjunto_features": str(conjunto),
            "slug": slug,
            "n_predicoes_externas": int(len(grupo)),
        })
        partes.append(
            grupo[
                [
                    "fold",
                    "grupo_externo",
                    "nome_arquivo",
                    "prob_contaminada",
                    "predicao",
                ]
            ].rename(
                columns={
                    "prob_contaminada": f"prob_{slug}",
                    "predicao": f"predito_0_50_{slug}",
                }
            )
        )

    if not modelos:
        raise ValueError(
            "Nenhum modelo visual possui cobertura externa completa para triagem."
        )

    return partes, modelos


def criar_tabela_integrada(
    folds: pd.DataFrame,
    predicoes: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    base = folds[folds["papel_amostra"].astype(str) == "teste_externo"].copy()
    base = base.rename(columns={"classe": "classe_real"})
    base = base.drop_duplicates(["fold", "grupo_externo", "nome_arquivo"])
    base = base[
        [
            "fold",
            "grupo_externo",
            "grupo_validacao",
            "nome_arquivo",
            "caminho_relativo",
            "classe_real",
            "alvo",
            "split_original",
            "experimento_tratamento",
        ]
    ].copy()

    partes, modelos = selecionar_predicoes_visuais(predicoes, len(base))
    tabela = base.copy()
    for parte in partes:
        tabela = tabela.merge(
            parte,
            on=["fold", "grupo_externo", "nome_arquivo"],
            how="left",
            validate="one_to_one",
        )

    colunas_prob = [f"prob_{item['slug']}" for item in modelos]
    faltantes = tabela[colunas_prob].isna().sum()
    if int(faltantes.sum()) != 0:
        raise ValueError(f"Ha probabilidades ausentes apos integracao: {faltantes.to_dict()}")

    tabela["modelos_visuais_completos"] = ",".join(item["modelo"] for item in modelos)
    tabela["estrategia_oficial_pre_especificada"] = "consenso_pre_especificado"
    tabela["criterio_definido_antes_avaliacao"] = True
    tabela["usa_resultado_externo_para_selecao"] = False
    return tabela, modelos


def filtrar_thresholds_modelos(thresholds: pd.DataFrame, modelos: list[dict]) -> pd.DataFrame:
    chaves = {
        (item["modelo"], item["conjunto_features"])
        for item in modelos
    }
    trabalho = thresholds.copy()
    mascara = trabalho.apply(
        lambda linha: (
            str(linha.get("modelo")),
            str(linha.get("conjunto_features")),
        )
        in chaves,
        axis=1,
    )
    return trabalho[mascara].copy()


def main() -> None:
    print("=" * 70)
    print("CRIANDO TABELA INTEGRADA DA TRIAGEM")
    print("=" * 70)

    folds = ler_csv_obrigatorio(CAMINHO_FOLDS)
    predicoes = ler_csv_obrigatorio(CAMINHO_PREDICOES_VALIDACAO)
    thresholds = ler_csv_obrigatorio(CAMINHO_THRESHOLDS_VALIDACAO)

    tabela, modelos = criar_tabela_integrada(folds, predicoes)
    thresholds_modelos = filtrar_thresholds_modelos(thresholds, modelos)

    gravar_csv_atomico(tabela, CAMINHO_TABELA_INTEGRADA)
    gravar_csv_atomico(thresholds_modelos, CAMINHO_THRESHOLDS_INTERNOS)
    gravar_json_atomico(
        {
            "protocolo": "triagem_preventiva_crossfit",
            "origem_predicoes": caminho_relativo(CAMINHO_PREDICOES_VALIDACAO),
            "origem_thresholds": caminho_relativo(CAMINHO_THRESHOLDS_VALIDACAO),
            "cenario_probabilidade": CENARIO_PROBABILIDADE,
            "modelos_visuais_com_cobertura_externa_completa": modelos,
            "estrategia_oficial": "consenso_pre_especificado",
            "criterio_definido_antes_avaliacao": True,
            "usa_resultado_externo_para_selecao": False,
            "exclui_media_maximo_por_falta_probabilidades_internas_amostra": True,
            "arquivos_saida": {
                "tabela_integrada": caminho_relativo(CAMINHO_TABELA_INTEGRADA),
                "thresholds_internos": caminho_relativo(CAMINHO_THRESHOLDS_INTERNOS),
            },
        },
        CAMINHO_MANIFESTO,
    )

    print(f"Amostras integradas: {len(tabela)}")
    print("Modelos visuais completos:")
    for modelo in modelos:
        print(f"- {modelo['modelo']} ({modelo['conjunto_features']})")
    print("Arquivos gerados:")
    print(f"- {CAMINHO_TABELA_INTEGRADA}")
    print(f"- {CAMINHO_THRESHOLDS_INTERNOS}")
    print(f"- {CAMINHO_MANIFESTO}")


if __name__ == "__main__":
    main()
