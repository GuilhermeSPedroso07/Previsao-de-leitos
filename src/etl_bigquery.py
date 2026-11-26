"""Extração e cache local/Colab dos dados do DATASUS, IBGE, SEDEC e INMET via BigQuery."""

import os
from pathlib import Path
import sys
from typing import Dict, Optional

from dotenv import load_dotenv
from google.cloud import bigquery
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = Path(os.getenv("DATA_DIR") or (PROJECT_ROOT / "data"))

_COLAB_AUTHENTICATED = False

MAPA_REGIOES_HIDRO: Dict[str, str] = {
    "Porto Alegre": "Metropolitana e Bacia do Guaíba",
    "Santa Cruz do Sul - Lajeado": "Vales (Taquari e Rio Pardo)",
    "Caxias do Sul": "Serra Gaúcha (Alto Taquari/Caí)",
    "Pelotas": "Região Sul e Lagoa dos Patos",
    "Passo Fundo": "Planalto Norte (Alto Jacuí/Uruguai)",
    "Santa Maria": "Depressão Central (Médio Jacuí)",
    "Ijuí": "Noroeste e Missões",
    "Uruguaiana": "Fronteira Oeste e Campanha",
}


def is_colab_env() -> bool:
    """Detecta se o código está rodando no ambiente Google Colab."""
    if "google.colab" in sys.modules:
        return True
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Lê segredo do .env/variável de ambiente local ou dos Secrets do Colab."""
    load_dotenv(PROJECT_ROOT / ".env")
    val = os.getenv(key)
    if (not val or not val.strip()) and is_colab_env():
        try:
            from google.colab import userdata

            val = userdata.get(key)
        except Exception:
            val = None
    if val is None:
        return default
    cleaned = val.strip().strip('"').strip("'")
    return cleaned if cleaned else default


def ensure_gcp_auth() -> None:
    """Executa autenticação interativa apenas no Colab; localmente usa ADC."""
    global _COLAB_AUTHENTICATED
    if not is_colab_env() or _COLAB_AUTHENTICATED:
        return
    from google.colab import auth

    auth.authenticate_user()
    _COLAB_AUTHENTICATED = True
    print("[COLAB] Autenticado com sucesso no Google Cloud.")


def get_gcp_project_id() -> str:
    """Obtém o ID do projeto GCP a partir do .env local, ambiente ou Secrets do Colab."""
    pid = get_secret("GCP_PROJECT_ID")
    if not pid:
        raise ValueError(
            "Defina 'GCP_PROJECT_ID' no arquivo .env na raiz do projeto, nos Secrets do Colab "
            "ou como variável de ambiente, e autentique via ADC ('gcloud auth application-default login')."
        )
    return pid


def get_bigquery_client(project_id: Optional[str] = None) -> bigquery.Client:
    """Inicializa o cliente BigQuery usando ADC local ou autenticação do Colab."""
    ensure_gcp_auth()
    resolved_project = project_id or get_gcp_project_id()
    return bigquery.Client(project=resolved_project)


def _load_parquet_cache(cache_path: Path | str, require_non_empty: bool = False) -> Optional[pd.DataFrame]:
    """Carrega cache Parquet se existir; remove caches vazios quando exigido."""
    path = Path(cache_path)
    if not path.exists():
        return None
    df_cached = pd.read_parquet(path)
    if require_non_empty and len(df_cached) == 0:
        path.unlink()
        return None
    return df_cached


def _save_parquet_cache(df: pd.DataFrame, cache_path: Path | str) -> None:
    """Garante o diretório de destino e persiste o DataFrame em Parquet."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def fetch_rs_hydro_regions(
    project_id: str,
    cache_path: Path | str = DATA_DIR / "rs_regioes_hidro_ibge.parquet",
) -> pd.DataFrame:
    """Extrai os municípios do RS e os mapeia para as 8 regiões hidrográficas."""
    cached = _load_parquet_cache(cache_path)
    if cached is not None:
        return cached

    client = get_bigquery_client(project_id)
    sql = """
    SELECT
        TRIM(CAST(id_municipio AS STRING)) AS id_municipio,
        nome AS nome_municipio,
        nome_regiao_intermediaria AS regiao_ibge
    FROM `basedosdados.br_bd_diretorios_brasil.municipio`
    WHERE sigla_uf = 'RS'
    """
    df = client.query(sql).to_dataframe()
    df["regiao_hidro"] = df["regiao_ibge"].map(MAPA_REGIOES_HIDRO).fillna(df["regiao_ibge"])
    _save_parquet_cache(df, cache_path)
    return df


def fetch_sedec_rs_complete(
    project_id: str,
    cache_path: Path | str = DATA_DIR / "sedec_rs_497_2010_2025.parquet",
) -> pd.DataFrame:
    """Extrai os decretos de desastres hidrológicos da SEDEC no RS (2010–2025)."""
    cached = _load_parquet_cache(cache_path)
    if cached is not None:
        return cached

    client = get_bigquery_client(project_id)
    sql = """
    SELECT DISTINCT
        SUBSTR(REGEXP_REPLACE(CAST(id_municipio AS STRING), r'[^0-9]', ''), 1, 7) AS id_municipio,
        DATE(COALESCE(data_decreto, data_portaria)) AS data_decreto
    FROM `basedosdados.br_sedec_desastres.reconhecimentos_realizados`
    WHERE COALESCE(data_decreto, data_portaria) IS NOT NULL
      AND SUBSTR(REGEXP_REPLACE(CAST(id_municipio AS STRING), r'[^0-9]', ''), 1, 2) = '43'
      AND EXTRACT(YEAR FROM COALESCE(data_decreto, data_portaria)) BETWEEN 2010 AND 2025
      AND (
          REGEXP_REPLACE(CAST(id_cobrade AS STRING), r'[^0-9]', '') LIKE '12%'
          OR REGEXP_REPLACE(CAST(id_cobrade AS STRING), r'[^0-9]', '') IN ('13214', '13215')
      )
    """
    cfg = bigquery.QueryJobConfig(maximum_bytes_billed=1000 * (1024**2), use_query_cache=True)
    df = client.query(sql, job_config=cfg).to_dataframe()
    df["data_decreto"] = pd.to_datetime(df["data_decreto"]).dt.normalize()
    _save_parquet_cache(df, cache_path)
    return df


def fetch_inmet_rs_wind(
    project_id: str,
    cache_path: Path | str = DATA_DIR / "inmet_rs_vento_2010_2025.parquet",
    ano_inicio: int = 2010,
    ano_fim: int = 2025,
) -> pd.DataFrame:
    """Extrai direção e velocidade do vento (INMET) no RS com foco no represamento na Barra de Rio Grande."""
    cached = _load_parquet_cache(cache_path, require_non_empty=True)
    if cached is not None:
        cached["data"] = pd.to_datetime(cached["data"]).dt.normalize()
        return cached

    client = get_bigquery_client(project_id)
    micro_table = "basedosdados.br_inmet_bdmep.microdados"
    est_table = "basedosdados.br_inmet_bdmep.estacao"

    micro_schema = {f.name: f.field_type for f in client.get_table(micro_table).schema}
    est_schema = {f.name: f.field_type for f in client.get_table(est_table).schema}
    micro_cols = list(micro_schema.keys())
    est_cols = list(est_schema.keys())

    est_id_micro = next(c for c in ["id_estacao", "codigo_estacao"] if c in micro_cols)
    est_id_est = next(c for c in ["id_estacao", "codigo_estacao"] if c in est_cols)
    mun_col = next(c for c in ["id_municipio", "codigo_municipio"] if c in est_cols)
    uf_col = next((c for c in ["sigla_uf", "uf"] if c in est_cols), None)
    data_col = next(c for c in ["data", "data_medicao"] if c in micro_cols)

    vel_col = next(
        (c for c in ["vento_velocidade", "velocidade_vento"] if c in micro_cols),
        next(c for c in micro_cols if "vento" in c and "veloc" in c),
    )
    dir_col = next(
        (c for c in ["vento_direcao", "direcao_vento"] if c in micro_cols),
        next(c for c in micro_cols if "vento" in c and "direc" in c),
    )
    raj_col = next(
        (c for c in ["vento_rajada_max", "rajada_max"] if c in micro_cols),
        next((c for c in micro_cols if "rajada" in c), vel_col),
    )

    ano_filter = (
        f"m.ano BETWEEN '{ano_inicio}' AND '{ano_fim}'"
        if micro_schema.get("ano") == "STRING"
        else f"m.ano BETWEEN {ano_inicio} AND {ano_fim}"
    )
    uf_filter = (
        f"e.{uf_col} = 'RS'"
        if uf_col and est_schema.get(uf_col) == "STRING"
        else f"SUBSTR(REGEXP_REPLACE(CAST(e.{mun_col} AS STRING), r'[^0-9]', ''), 1, 2) = '43'"
    )
    dt_expr = (
        f"COALESCE(SAFE.PARSE_DATE('%Y-%m-%d', m.{data_col}), SAFE.PARSE_DATE('%Y%m%d', m.{data_col}))"
        if micro_schema.get(data_col) == "STRING"
        else f"SAFE_CAST(m.{data_col} AS DATE)"
    )

    # Decomposição vetorial horária antes da média diária evita erro de média circular em graus.
    # Na Barra de Rio Grande (A802 / 4315602), o eixo SSW (~200°) represa o canal da Lagoa dos Patos.
    sql = f"""
    WITH vento_horario AS (
        SELECT
            SUBSTR(REGEXP_REPLACE(CAST(e.{mun_col} AS STRING), r'[^0-9]', ''), 1, 7) AS id_municipio,
            TRIM(CAST(m.{est_id_micro} AS STRING)) AS id_estacao,
            {dt_expr} AS data,
            SAFE_CAST(m.{vel_col} AS FLOAT64) AS v_ms,
            SAFE_CAST(m.{dir_col} AS FLOAT64) AS dir_deg,
            SAFE_CAST(m.{raj_col} AS FLOAT64) AS raj_ms
        FROM `{micro_table}` m
        INNER JOIN `{est_table}` e
            ON TRIM(CAST(m.{est_id_micro} AS STRING)) = TRIM(CAST(e.{est_id_est} AS STRING))
        WHERE {ano_filter}
          AND {uf_filter}
          AND SAFE_CAST(m.{vel_col} AS FLOAT64) IS NOT NULL
          AND SAFE_CAST(m.{dir_col} AS FLOAT64) IS NOT NULL
    )
    SELECT
        id_municipio,
        id_estacao,
        data,
        AVG(v_ms) AS vento_velocidade_ms,
        MAX(COALESCE(raj_ms, v_ms)) AS vento_rajada_max_ms,
        AVG(-v_ms * COS(dir_deg * ACOS(-1) / 180.0)) AS vento_sul_ms,
        AVG(-v_ms * SIN(dir_deg * ACOS(-1) / 180.0)) AS vento_oeste_ms,
        AVG(GREATEST(0.0, v_ms * COS((dir_deg - 200.0) * ACOS(-1) / 180.0))) AS vento_represamento_barra_ms,
        AVG(v_ms * COS((dir_deg - 200.0) * ACOS(-1) / 180.0)) AS vento_escoamento_barra_ms,
        MAX(CASE WHEN id_estacao = 'A802' OR id_municipio = '4315602' THEN 1 ELSE 0 END) = 1 AS is_barra_rg
    FROM vento_horario
    WHERE data IS NOT NULL
    GROUP BY 1, 2, 3
    ORDER BY 1, 3
    """
    cfg = bigquery.QueryJobConfig(maximum_bytes_billed=6 * (1024**3), use_query_cache=True)
    df = client.query(sql, job_config=cfg).to_dataframe()
    df["data"] = pd.to_datetime(df["data"]).dt.normalize()
    _save_parquet_cache(df, cache_path)
    print(f"[INMET VENTO RS] Extraídas {len(df):,} linhas estação-dia (cache: {cache_path}).")
    return df


def fetch_cnes_rs_sus_monthly(
    project_id: str,
    cache_path: Path | str = DATA_DIR / "cnes_rs_leitos_sus_2011_2024.parquet",
) -> pd.DataFrame:
    """Extrai oferta mensal do CNES no RS filtrada estritamente para leitos SUS (Total e UTI)."""
    cached = _load_parquet_cache(cache_path, require_non_empty=True)
    if cached is not None:
        return cached

    client = get_bigquery_client(project_id)
    table_id = "basedosdados.br_ms_cnes.leito"
    schema_map = {f.name: f.field_type for f in client.get_table(table_id).schema}
    schema_cols = list(schema_map.keys())

    mun_col = next(c for c in ["id_municipio", "id_municipio_estabelecimento"] if c in schema_cols)
    sus_col = next(c for c in schema_cols if "sus" in c and "nao" not in c and "quantidade" in c)
    tipo_col = next(c for c in schema_cols if "tipo_leito" in c)
    uf_filter = (
        "sigla_uf = 'RS'"
        if schema_map.get("sigla_uf") == "STRING"
        else "SUBSTR(CAST(id_municipio AS STRING), 1, 2) = '43'"
    )
    ano_filter = "ano BETWEEN '2011' AND '2024'" if schema_map.get("ano") == "STRING" else "ano BETWEEN 2011 AND 2024"

    sql = f"""
    SELECT
        SUBSTR(REGEXP_REPLACE(CAST({mun_col} AS STRING), r'[^0-9]', ''), 1, 6) AS id_mun_6,
        CAST(ano AS INT64) AS ano,
        CAST(mes AS INT64) AS mes,
        SUM(COALESCE(SAFE_CAST({sus_col} AS FLOAT64), 0)) AS oferta_total_sus,
        SUM(CASE WHEN REGEXP_REPLACE(CAST({tipo_col} AS STRING), r'^0+', '') = '3'
                 THEN COALESCE(SAFE_CAST({sus_col} AS FLOAT64), 0) ELSE 0 END) AS oferta_uti_sus
    FROM `{table_id}`
    WHERE {uf_filter} AND {ano_filter}
    GROUP BY 1, 2, 3
    """
    cfg = bigquery.QueryJobConfig(maximum_bytes_billed=4 * (1024**3), use_query_cache=True)
    df = client.query(sql, job_config=cfg).to_dataframe()
    _save_parquet_cache(df, cache_path)
    return df


def fetch_sih_rs_sem_dengue(
    project_id: str,
    cache_path: Path | str = DATA_DIR / "sih_rs_urgencia_cids_sem_dengue_2011_2024.parquet",
) -> pd.DataFrame:
    """Extrai o SIH/SUS do RS deduplicado por AIH e agrupado por data_internacao, sem Dengue (A90–A92)."""
    cached = _load_parquet_cache(cache_path, require_non_empty=True)
    if cached is not None:
        return cached

    client = get_bigquery_client(project_id)
    table_id = "basedosdados.br_ms_sih.aihs_reduzidas"
    schema_map = {f.name: f.field_type for f in client.get_table(table_id).schema}
    schema_cols = list(schema_map.keys())

    mun_est_col = next(c for c in ["id_municipio_estabelecimento", "id_municipio"] if c in schema_cols)
    aih_col = next((c for c in ["id_aih", "numero_aih", "sequencial_aih"] if c in schema_cols), None)
    dt_int_col = next((c for c in ["data_internacao", "dt_inter"] if c in schema_cols), None)
    dt_sai_col = next((c for c in ["data_saida", "dt_saida"] if c in schema_cols), None)
    carater_col = next(c for c in schema_cols if "carater" in c)
    cid_col = next(
        (c for c in ["diagnostico_principal", "cid_principal", "cid_principal_categoria", "diag_princ"] if c in schema_cols),
        next(c for c in schema_cols if ("diagnostico" in c or "cid" in c) and "princ" in c),
    )

    ano_filter = "ano BETWEEN '2011' AND '2025'" if schema_map.get("ano") == "STRING" else "ano BETWEEN 2011 AND 2025"
    dt_expr = (
        f"COALESCE(SAFE.PARSE_DATE('%Y-%m-%d', {dt_int_col}), SAFE.PARSE_DATE('%Y%m%d', {dt_int_col}))"
        if schema_map.get(dt_int_col) == "STRING"
        else f"SAFE_CAST({dt_int_col} AS DATE)"
    )

    urg_expr = (
        f"CASE WHEN REGEXP_REPLACE(TRIM(CAST({carater_col} AS STRING)), r'^0+', '') "
        f"IN ('2', '3', '4', '5', '6') THEN 1 ELSE 0 END"
    )

    # SEM DENGUE (A90–A92): apenas Leptospirose (A27), Hídricas/Diarreia (A00–A09, B15),
    # Respiratórias Agudas/Asma (J00–J22, J45–J46), Pele (L00–L08) e Traumas/Hipotermia (S00–T98)
    cid_regex = r"^(A27|A0[0-9]|B15|J0[0-9]|J1[0-8]|J2[0-2]|J4[5-6]|L0[0-8]|S[0-9]{2}|T[0-9]{2})"
    cid_expr = f"CASE WHEN REGEXP_CONTAINS(UPPER(TRIM(CAST({cid_col} AS STRING))), r'{cid_regex}') THEN 1 ELSE 0 END"

    dedup_cols = [
        f"SUBSTR(REGEXP_REPLACE(CAST({mun_est_col} AS STRING), r'[^0-9]', ''), 1, 6) AS id_mun_6",
        f"{dt_expr} AS dt_int",
        "COALESCE(SAFE_CAST(quantidade_dias_permanencia AS FLOAT64), 0) AS dias_tot",
        "COALESCE(SAFE_CAST(quantidade_dias_uti_mes AS FLOAT64), 0) AS dias_uti",
        f"{urg_expr} AS is_urg",
        f"{cid_expr} AS is_cid_ench",
    ]
    if aih_col:
        dedup_cols.append(f"CAST({aih_col} AS STRING) AS aih_id")
    if dt_sai_col:
        dedup_cols.append(f"CAST({dt_sai_col} AS STRING) AS dt_sai")

    sql = f"""
    WITH base_dedup AS (
        SELECT DISTINCT {", ".join(dedup_cols)}
        FROM `{table_id}`
        WHERE {ano_filter}
          AND SUBSTR(REGEXP_REPLACE(CAST({mun_est_col} AS STRING), r'[^0-9]', ''), 1, 2) = '43'
    )
    SELECT
        id_mun_6,
        EXTRACT(YEAR FROM dt_int) AS ano,
        EXTRACT(MONTH FROM dt_int) AS mes,
        COUNT(*) AS internacoes_geral,
        SUM(is_urg) AS internacoes_urgencia,
        SUM(is_cid_ench) AS internacoes_cid_enchente,
        SUM(CASE WHEN is_urg = 1 THEN dias_tot ELSE 0 END) AS diarias_urg_total,
        SUM(CASE WHEN is_urg = 1 THEN dias_uti ELSE 0 END) AS diarias_urg_uti,
        SUM(CASE WHEN is_cid_ench = 1 THEN dias_tot ELSE 0 END) AS diarias_cid_total,
        SUM(CASE WHEN is_cid_ench = 1 THEN dias_uti ELSE 0 END) AS diarias_cid_uti
    FROM base_dedup
    WHERE dt_int IS NOT NULL
      AND EXTRACT(YEAR FROM dt_int) BETWEEN 2011 AND 2024
    GROUP BY 1, 2, 3
    """
    cfg = bigquery.QueryJobConfig(maximum_bytes_billed=15 * (1024**3), use_query_cache=True)
    df_sih = client.query(sql, job_config=cfg).to_dataframe()
    _save_parquet_cache(df_sih, cache_path)

    print("[DIAGNÓSTICO 2024 SEM DENGUE — INTERNAÇÕES POR MÊS]")
    print(df_sih[df_sih["ano"] == 2024].groupby("mes")[["internacoes_urgencia", "internacoes_cid_enchente"]].sum().to_string())
    return df_sih
