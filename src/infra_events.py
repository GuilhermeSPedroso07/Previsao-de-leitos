"""Catálogo e expansão diária de falhas binárias de infraestrutura (diques e casas de bombas)."""

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.etl_bigquery import DATA_DIR

INFRA_CSV_COLUMNS: List[str] = [
    "regiao_hidro",
    "data_inicio",
    "data_fim",
    "falha_dique",
    "falha_casa_bombas",
    "municipio_ref",
    "evento",
    "fonte",
]

# Catálogo auditável com episódios históricos documentados (treino pré-2024 + crise de maio/2024)
DEFAULT_INFRA_EVENTS: List[Dict[str, object]] = [
    {
        "regiao_hidro": "Metropolitana e Bacia do Guaíba",
        "data_inicio": "2015-10-12",
        "data_fim": "2015-10-20",
        "falha_dique": 0,
        "falha_casa_bombas": 1,
        "municipio_ref": "Porto Alegre / São Leopoldo",
        "evento": "Fechamento de comportas e sobrecarga/desligamento de bombas na cheia do Guaíba e Sinos",
        "fonte": "Defesa Civil RS / DMAE (out/2015)",
    },
    {
        "regiao_hidro": "Vales (Taquari e Rio Pardo)",
        "data_inicio": "2023-09-04",
        "data_fim": "2023-09-10",
        "falha_dique": 1,
        "falha_casa_bombas": 1,
        "municipio_ref": "Muçum / Roca Sales / Lajeado / Estrela",
        "evento": "Galgamento de contenções ribeirinhas e colapso elétrico/bombeamento no Vale do Taquari",
        "fonte": "Defesa Civil RS / SGB (set/2023)",
    },
    {
        "regiao_hidro": "Metropolitana e Bacia do Guaíba",
        "data_inicio": "2023-09-25",
        "data_fim": "2023-09-30",
        "falha_dique": 0,
        "falha_casa_bombas": 1,
        "municipio_ref": "Porto Alegre / Eldorado do Sul",
        "evento": "Fechamento de comportas do Guaíba e falha parcial em casas de bombas",
        "fonte": "DMAE / Defesa Civil RS (set/2023)",
    },
    {
        "regiao_hidro": "Metropolitana e Bacia do Guaíba",
        "data_inicio": "2023-11-18",
        "data_fim": "2023-11-25",
        "falha_dique": 1,
        "falha_casa_bombas": 1,
        "municipio_ref": "Porto Alegre / Canoas / São Leopoldo",
        "evento": "Extravasamento em diques secundários e desligamento preventivo de bombeamento",
        "fonte": "Defesa Civil RS (nov/2023)",
    },
    {
        "regiao_hidro": "Metropolitana e Bacia do Guaíba",
        "data_inicio": "2024-05-03",
        "data_fim": "2024-05-18",
        "falha_dique": 1,
        "falha_casa_bombas": 1,
        "municipio_ref": "Porto Alegre",
        "evento": "Rompimento da Comporta 14 e desligamento/alagamento de casas de bombas do DMAE",
        "fonte": "DMAE / Defesa Civil Porto Alegre (mai/2024)",
    },
    {
        "regiao_hidro": "Metropolitana e Bacia do Guaíba",
        "data_inicio": "2024-05-04",
        "data_fim": "2024-05-20",
        "falha_dique": 1,
        "falha_casa_bombas": 1,
        "municipio_ref": "Canoas",
        "evento": "Rompimento dos diques Mathias Velho e Rio Branco e submersão das casas de bombas",
        "fonte": "Prefeitura de Canoas / Defesa Civil RS (mai/2024)",
    },
    {
        "regiao_hidro": "Metropolitana e Bacia do Guaíba",
        "data_inicio": "2024-05-04",
        "data_fim": "2024-05-18",
        "falha_dique": 1,
        "falha_casa_bombas": 1,
        "municipio_ref": "São Leopoldo",
        "evento": "Rompimento do dique da Rua João Corrêa e pane nas casas de bombas (Rodoviária/Ginásio)",
        "fonte": "Prefeitura de São Leopoldo / Defesa Civil RS (mai/2024)",
    },
    {
        "regiao_hidro": "Região Sul e Lagoa dos Patos",
        "data_inicio": "2024-05-08",
        "data_fim": "2024-05-22",
        "falha_dique": 1,
        "falha_casa_bombas": 1,
        "municipio_ref": "Pelotas / Rio Grande",
        "evento": "Extravasamento/ruptura de diques de proteção e sobrecarga/desligamento de bombeamento na Lagoa dos Patos",
        "fonte": "Defesa Civil RS / Prefeituras de Pelotas e Rio Grande (mai/2024)",
    },
]


def ensure_infra_events_csv(csv_path: Path | str = DATA_DIR / "eventos_infraestrutura_rs.csv") -> Path:
    """Cria o catálogo CSV de falhas de infraestrutura caso não exista."""
    path = Path(csv_path)
    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INFRA_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(DEFAULT_INFRA_EVENTS)
    return path


def load_infra_events_daily(
    dates: pd.DatetimeIndex,
    regioes: List[str],
    csv_path: Path | str = DATA_DIR / "eventos_infraestrutura_rs.csv",
) -> pd.DataFrame:
    """Expande eventos de rompimento de diques e desligamento de bombas para grade diária binária."""
    path = ensure_infra_events_csv(csv_path)
    df_ev = pd.read_csv(path)

    all_dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values()
    full_idx = pd.MultiIndex.from_product([sorted(regioes), all_dates], names=["regiao_hidro", "data"])
    daily = pd.DataFrame(index=full_idx).reset_index()
    daily["falha_dique"] = np.float32(0.0)
    daily["falha_casa_bombas"] = np.float32(0.0)

    if df_ev.empty:
        daily["falha_infra_ativa"] = np.float32(0.0)
        return daily

    df_ev["data_inicio"] = pd.to_datetime(df_ev["data_inicio"]).dt.normalize()
    df_ev["data_fim"] = pd.to_datetime(df_ev["data_fim"]).dt.normalize()
    df_ev["falha_dique"] = pd.to_numeric(df_ev["falha_dique"], errors="coerce").fillna(0).clip(0, 1)
    df_ev["falha_casa_bombas"] = pd.to_numeric(df_ev["falha_casa_bombas"], errors="coerce").fillna(0).clip(0, 1)

    for _, row in df_ev.iterrows():
        reg = str(row["regiao_hidro"]).strip()
        if reg not in regioes:
            continue
        mask = (
            (daily["regiao_hidro"] == reg)
            & (daily["data"] >= row["data_inicio"])
            & (daily["data"] <= row["data_fim"])
        )
        if float(row["falha_dique"]) >= 1.0:
            daily.loc[mask, "falha_dique"] = np.float32(1.0)
        if float(row["falha_casa_bombas"]) >= 1.0:
            daily.loc[mask, "falha_casa_bombas"] = np.float32(1.0)

    daily["falha_dique"] = daily["falha_dique"].astype(np.float32)
    daily["falha_casa_bombas"] = daily["falha_casa_bombas"].astype(np.float32)
    daily["falha_infra_ativa"] = np.maximum(daily["falha_dique"], daily["falha_casa_bombas"]).astype(np.float32)
    return daily
