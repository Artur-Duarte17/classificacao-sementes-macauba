from pathlib import Path
import unicodedata
import pandas as pd


# ============================================================
# SCRIPT 02 - CRIAR RÓTULOS A PARTIR DAS PLANILHAS
# ------------------------------------------------------------
# Objetivo:
# - Ler as planilhas do projeto
# - Identificar se cada semente teve contaminação registrada
# - Criar a classe binária:
#     contaminada
#     nao_contaminada
# - Salvar um CSV com os rótulos consolidados
#
# Este script ainda NÃO copia imagens.
# Ele apenas cria a tabela de rótulos.
# ============================================================


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_PLANILHAS = PASTA_PROJETO / "dados_originais" / "planilhas"
PASTA_SAIDA = PASTA_PROJETO / "saidas" / "tabelas"


def normalizar_texto(texto: str) -> str:
    """
    Normaliza texto para comparação:
    - transforma em minúsculo
    - remove acentos
    - remove espaços duplicados
    """
    texto = str(texto).strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = " ".join(texto.split())
    return texto


def encontrar_planilha(palavras: list[str]) -> Path:
    """
    Procura uma planilha cujo nome contenha todas as palavras informadas.
    Isso evita erro por causa de acento ou nome ligeiramente diferente.
    """
    arquivos = list(PASTA_PLANILHAS.glob("*.xlsx")) + list(PASTA_PLANILHAS.glob("*.xls"))

    for arquivo in arquivos:
        nome_normalizado = normalizar_texto(arquivo.name)

        if all(normalizar_texto(palavra) in nome_normalizado for palavra in palavras):
            return arquivo

    raise FileNotFoundError(f"Nenhuma planilha encontrada contendo: {palavras}")


def limpar_id(valor) -> str:
    """
    Padroniza o ID da semente.
    Exemplos:
    - 'a1 ' -> 'a1'
    - 1.0 -> '1'
    - '001' -> '001'
    """
    if pd.isna(valor):
        return ""

    texto = str(valor).strip().lower()

    # Remove .0 quando o Excel transforma número inteiro em decimal
    if texto.endswith(".0"):
        texto = texto[:-2]

    return texto


def id_com_zeros(valor, tamanho=3) -> str:
    """
    Converte IDs numéricos para o padrão das imagens do Micro-ondas.
    Exemplo:
    1 -> 001
    12 -> 012
    123 -> 123
    """
    texto = limpar_id(valor)

    if texto.isdigit():
        return texto.zfill(tamanho)

    return texto


def classe_por_contaminacao(valor_contaminacao) -> str:
    """
    Define a classe final binária.
    """
    if int(valor_contaminacao) == 1:
        return "contaminada"

    return "nao_contaminada"


def processar_teste2() -> pd.DataFrame:
    """
    Processa a planilha TABELA PARA ANALISE - TESTE 2.xlsx.

    Estrutura esperada:
    - Trat
    - ID
    - Dias de Germ
    - Germ
    - Cont
    """

    caminho = encontrar_planilha(["teste 2"])
    print(f"Lendo TESTE 2: {caminho.name}")

    df = pd.read_excel(caminho)

    # Padroniza nomes das colunas
    df.columns = [str(c).strip() for c in df.columns]

    # Mantém só linhas úteis
    df = df.dropna(subset=["Trat", "ID"], how="any").copy()

    # Converte Germ e Cont para número
    df["Germ"] = pd.to_numeric(df["Germ"], errors="coerce").fillna(0)
    df["Cont"] = pd.to_numeric(df["Cont"], errors="coerce").fillna(0)

    # Agrupa por tratamento e ID.
    # Se contaminou em qualquer dia, contaminou = 1.
    agrupado = (
        df.groupby(["Trat", "ID"], dropna=False)
        .agg(
            contaminou=("Cont", "max"),
            germinou=("Germ", "max"),
            qtd_observacoes=("ID", "size")
        )
        .reset_index()
    )

    mapa_pastas = {
        "TC": "T - C1",
        "T2": "T - C2",
        "T3": "T3",
        "T4": "T4",
        "T5": "T5",
        "T6": "T6",
    }

    registros = []

    for _, linha in agrupado.iterrows():
        tratamento = str(linha["Trat"]).strip()
        id_original = limpar_id(linha["ID"])
        contaminou = int(linha["contaminou"] > 0)
        germinou = int(linha["germinou"] > 0)

        registros.append({
            "origem_planilha": caminho.name,
            "experimento": "TESTE 2",
            "tratamento_planilha": tratamento,
            "pasta_esperada": mapa_pastas.get(tratamento, tratamento),
            "id_semente_original": id_original,
            "id_busca": id_original,
            "contaminou": contaminou,
            "germinou": germinou,
            "classe": classe_por_contaminacao(contaminou),
            "qtd_observacoes": int(linha["qtd_observacoes"]),
            "observacao": ""
        })

    return pd.DataFrame(registros)


def processar_piloto() -> pd.DataFrame:
    """
    Processa a planilha Piloto -Contaminacao-Germinacao-Umidade.xlsx.

    A aba Contaminação tem uma estrutura especial:
    - a primeira linha tem datas
    - a segunda linha tem os nomes reais das colunas
    Por isso usamos header=1.
    """

    caminho = encontrar_planilha(["piloto"])
    print(f"Lendo Piloto: {caminho.name}")

    df = pd.read_excel(caminho, sheet_name="Contaminação", header=1)

    # Padroniza nomes
    df.columns = [str(c).strip() for c in df.columns]

    # Remove linhas sem ID
    df = df.dropna(subset=["ID"], how="any").copy()

    # Remove linhas que não são sementes
    df = df[df["ID"].astype(str).str.lower().str.strip() != "id"].copy()

    # Colunas de contaminação e germinação
    colunas_cont = [c for c in df.columns if c.startswith("Contaminação")]
    colunas_germ = [c for c in df.columns if c.startswith("Germinação")]

    for coluna in colunas_cont + colunas_germ:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce").fillna(0)

    registros = []

    for _, linha in df.iterrows():
        id_original = limpar_id(linha["ID"])
        tratamento = str(linha["Tratamento"]).strip()

        contaminou = int(linha[colunas_cont].max() > 0)
        germinou = int(linha[colunas_germ].max() > 0)

        registros.append({
            "origem_planilha": caminho.name,
            "experimento": "Piloto",
            "tratamento_planilha": tratamento,
            "pasta_esperada": "Piloto",
            "id_semente_original": id_original,
            "id_busca": id_original,
            "contaminou": contaminou,
            "germinou": germinou,
            "classe": classe_por_contaminacao(contaminou),
            "qtd_observacoes": len(colunas_cont),
            "observacao": ""
        })

    return pd.DataFrame(registros)


def processar_novos_indices() -> pd.DataFrame:
    """
    Processa a planilha 23-04-2026 - Novos Índices.xlsx.

    Essa planilha já tem colunas:
    - ID
    - Tratamento
    - Contaminação D7
    - Contaminação D14
    - Contaminação D28
    """

    caminho = encontrar_planilha(["novos", "indices"])
    print(f"Lendo Novos Índices: {caminho.name}")

    df = pd.read_excel(caminho)

    # Remove espaços extras nos nomes das colunas
    df.columns = [str(c).strip() for c in df.columns]

    # Colunas de contaminação
    colunas_cont = [c for c in df.columns if c.startswith("Contaminação")]

    for coluna in colunas_cont:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce").fillna(0)

    registros = []

    for _, linha in df.iterrows():
        id_original = limpar_id(linha["ID"])
        id_busca = id_com_zeros(linha["ID"], tamanho=3)

        tratamento = str(linha["Tratamento"]).strip()
        contaminou = int(linha[colunas_cont].max() > 0)

        registros.append({
            "origem_planilha": caminho.name,
            "experimento": "Micro-ondas",
            "tratamento_planilha": tratamento,
            "pasta_esperada": "Micro-ondas",
            "id_semente_original": id_original,
            "id_busca": id_busca,
            "contaminou": contaminou,
            "germinou": "",
            "classe": classe_por_contaminacao(contaminou),
            "qtd_observacoes": len(colunas_cont),
            "observacao": "Rótulo derivado das colunas de contaminação D7/D14/D28"
        })

    return pd.DataFrame(registros)


def main():
    print("=" * 60)
    print("CRIANDO RÓTULOS DAS PLANILHAS")
    print("=" * 60)

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    tabelas = []

    # Processa cada fonte de dados
    tabelas.append(processar_teste2())
    tabelas.append(processar_piloto())
    tabelas.append(processar_novos_indices())

    rotulos = pd.concat(tabelas, ignore_index=True)

    # Garante tipo inteiro na coluna contaminou
    rotulos["contaminou"] = rotulos["contaminou"].astype(int)

    # Salva tabela completa de rótulos
    caminho_rotulos = PASTA_SAIDA / "rotulos_planilhas.csv"
    rotulos.to_csv(caminho_rotulos, index=False, encoding="utf-8-sig")

    # Resumo geral
    resumo = (
        rotulos.groupby(["experimento", "pasta_esperada", "classe"])
        .size()
        .reset_index(name="quantidade")
        .sort_values(["experimento", "pasta_esperada", "classe"])
    )

    caminho_resumo = PASTA_SAIDA / "resumo_rotulos.csv"
    resumo.to_csv(caminho_resumo, index=False, encoding="utf-8-sig")

    # Verifica possíveis duplicatas de chave
    # A chave esperada é: experimento + pasta + id da imagem
    duplicatas = (
        rotulos.groupby(["experimento", "pasta_esperada", "id_busca"])
        .size()
        .reset_index(name="quantidade")
    )

    duplicatas = duplicatas[duplicatas["quantidade"] > 1].copy()

    caminho_duplicatas = PASTA_SAIDA / "duplicatas_rotulos.csv"
    duplicatas.to_csv(caminho_duplicatas, index=False, encoding="utf-8-sig")

    print()
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(f"Total de rótulos criados: {len(rotulos)}")
    print(f"Contaminadas: {(rotulos['classe'] == 'contaminada').sum()}")
    print(f"Não contaminadas: {(rotulos['classe'] == 'nao_contaminada').sum()}")
    print(f"Duplicatas de rótulo: {len(duplicatas)}")

    print()
    print("Resumo por experimento:")
    print(resumo.to_string(index=False))

    print()
    print("Arquivos gerados:")
    print(f"- {caminho_rotulos}")
    print(f"- {caminho_resumo}")
    print(f"- {caminho_duplicatas}")

    print()
    print("Criação de rótulos concluída.")


if __name__ == "__main__":
    main()