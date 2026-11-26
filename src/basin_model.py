"""Estágio 1: Engenharia hidrológica regional, roteamento Montante->Jusante e HydroFiLMRegionalBasinNet."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Grafo causal hidrológico (Montante -> Jusante)
UPSTREAM_BASINS: Dict[str, List[str]] = {
    "Serra Gaúcha (Alto Taquari/Caí)": [],
    "Planalto Norte (Alto Jacuí/Uruguai)": [],
    "Vales (Taquari e Rio Pardo)": [
        "Serra Gaúcha (Alto Taquari/Caí)",
        "Planalto Norte (Alto Jacuí/Uruguai)",
    ],
    "Depressão Central (Médio Jacuí)": [
        "Planalto Norte (Alto Jacuí/Uruguai)",
    ],
    "Metropolitana e Bacia do Guaíba": [
        "Serra Gaúcha (Alto Taquari/Caí)",
        "Vales (Taquari e Rio Pardo)",
        "Depressão Central (Médio Jacuí)",
        "Planalto Norte (Alto Jacuí/Uruguai)",
    ],
    "Região Sul e Lagoa dos Patos": [
        "Metropolitana e Bacia do Guaíba",
        "Vales (Taquari e Rio Pardo)",
        "Depressão Central (Médio Jacuí)",
    ],
    "Noroeste e Missões": [
        "Planalto Norte (Alto Jacuí/Uruguai)",
    ],
    "Fronteira Oeste e Campanha": [
        "Noroeste e Missões",
        "Depressão Central (Médio Jacuí)",
    ],
}

BASE_DYNAMIC_COLS: List[str] = [
    "precip_media_mm",
    "precip_max_reg_mm",
    "precip_max_horaria_mm",
    "precip_acum_3d",
    "precip_acum_7d",
    "precip_acum_15d",
    "precip_acum_30d",
    "api_indice_saturacao",
    "escoamento_efetivo_mm",
    "montante_chuva_7d",
    "montante_chuva_15d",
    "montante_chuva_30d",
    "montante_escoamento_7d",
    "montante_escoamento_15d",
    "temp_media_c",
    "temp_max_c",
    "umidade_rel_media",
    "pressao_media_hpa",
    "pressao_var_3d",
    "doy_sin",
    "doy_cos",
]

WIND_DYNAMIC_COLS: List[str] = [
    "vento_velocidade_ms",
    "vento_rajada_max_ms",
    "vento_sul_ms",
    "vento_oeste_ms",
    "barra_rg_vento_represamento_ms",
    "barra_rg_vento_represamento_3d",
    "barra_rg_vento_escoamento_ms",
]

ANA_DYNAMIC_COLS: List[str] = [
    "cota_max_cm",
    "cota_var_24h_cm",
    "cota_var_72h_cm",
    "montante_cota_max_cm",
]

INFRA_BINARY_COLS: List[str] = [
    "falha_dique",
    "falha_casa_bombas",
]

REG_STATIC_COLS: List[str] = [
    "indice_impermeabilidade",
    "indice_retencao_vegetal",
    "altitude_estacao",
    "latitude",
    "longitude",
]


def _merge_wind_features(reg_day: pd.DataFrame, df_wind: pd.DataFrame, df_reg: pd.DataFrame) -> pd.DataFrame:
    """Incorpora vento regional (INMET) e sinal de represamento na Barra de Rio Grande."""
    w = df_wind.copy()
    w["id_municipio"] = w["id_municipio"].astype(str).str.strip()
    w["data"] = pd.to_datetime(w["data"]).dt.normalize()
    w = w.merge(df_reg[["id_municipio", "regiao_hidro"]], on="id_municipio", how="inner")

    if not w.empty:
        w_reg = w.groupby(["regiao_hidro", "data"], as_index=False).agg(
            vento_velocidade_ms=("vento_velocidade_ms", "mean"),
            vento_rajada_max_ms=("vento_rajada_max_ms", "max"),
            vento_sul_ms=("vento_sul_ms", "mean"),
            vento_oeste_ms=("vento_oeste_ms", "mean"),
        )
        reg_day = reg_day.merge(w_reg, on=["regiao_hidro", "data"], how="left")
    else:
        for col in ("vento_velocidade_ms", "vento_rajada_max_ms", "vento_sul_ms", "vento_oeste_ms"):
            reg_day[col] = 0.0

    # Série costeira dedicada à Barra de Rio Grande (estação A802 / município 4315602)
    barra = w[w["is_barra_rg"].astype(bool)] if "is_barra_rg" in w.columns else pd.DataFrame()
    if barra.empty and not w.empty:
        barra = w[w["regiao_hidro"] == "Região Sul e Lagoa dos Patos"]

    if not barra.empty:
        barra_day = (
            barra.groupby("data", as_index=False)
            .agg(
                barra_rg_vento_represamento_ms=("vento_represamento_barra_ms", "mean"),
                barra_rg_vento_escoamento_ms=("vento_escoamento_barra_ms", "mean"),
            )
            .sort_values("data")
        )
        barra_day["barra_rg_vento_represamento_3d"] = (
            barra_day["barra_rg_vento_represamento_ms"].rolling(3, min_periods=1).mean()
        )
        reg_day = reg_day.merge(barra_day, on="data", how="left")
    else:
        reg_day["barra_rg_vento_represamento_ms"] = 0.0
        reg_day["barra_rg_vento_represamento_3d"] = 0.0
        reg_day["barra_rg_vento_escoamento_ms"] = 0.0

    for col in WIND_DYNAMIC_COLS:
        reg_day[col] = reg_day.groupby("regiao_hidro")[col].ffill().bfill().fillna(0.0)
    return reg_day


def _merge_ana_features(reg_day: pd.DataFrame, df_ana: pd.DataFrame, regioes: List[str]) -> pd.DataFrame:
    """Incorpora cotas fluviométricas da ANA e propaga cota de montante causalmente (shift 1)."""
    ana = df_ana.copy()
    ana["data"] = pd.to_datetime(ana["data"]).dt.normalize()
    ana_cols = ["regiao_hidro", "data", "cota_max_cm", "cota_var_24h_cm", "cota_var_72h_cm"]
    reg_day = reg_day.merge(ana[ana_cols].drop_duplicates(["regiao_hidro", "data"]), on=["regiao_hidro", "data"], how="left")

    reg_day["cota_max_cm"] = reg_day.groupby("regiao_hidro")["cota_max_cm"].ffill().bfill().fillna(0.0)
    reg_day["cota_var_24h_cm"] = reg_day["cota_var_24h_cm"].fillna(0.0)
    reg_day["cota_var_72h_cm"] = reg_day["cota_var_72h_cm"].fillna(0.0)

    piv_cota = reg_day.pivot(index="data", columns="regiao_hidro", values="cota_max_cm").shift(1).fillna(0.0)
    up_cota = {}
    for reg in regioes:
        ups = [u for u in UPSTREAM_BASINS.get(reg, []) if u in piv_cota.columns]
        up_cota[reg] = piv_cota[ups].max(axis=1) if ups else pd.Series(0.0, index=piv_cota.index)

    up_cota_df = pd.DataFrame(up_cota).stack().reset_index()
    up_cota_df.columns = ["data", "regiao_hidro", "montante_cota_max_cm"]
    reg_day = reg_day.merge(up_cota_df, on=["regiao_hidro", "data"], how="left")
    reg_day["montante_cota_max_cm"] = reg_day["montante_cota_max_cm"].fillna(0.0)
    return reg_day


def _merge_infra_features(reg_day: pd.DataFrame, df_infra_daily: pd.DataFrame) -> pd.DataFrame:
    """Incorpora indicadores binários de falha de diques e casas de bombas."""
    inf = df_infra_daily[["regiao_hidro", "data", "falha_dique", "falha_casa_bombas"]].copy()
    inf["data"] = pd.to_datetime(inf["data"]).dt.normalize()
    reg_day = reg_day.merge(inf.drop_duplicates(["regiao_hidro", "data"]), on=["regiao_hidro", "data"], how="left")
    for col in INFRA_BINARY_COLS:
        reg_day[col] = reg_day[col].fillna(0.0).clip(0.0, 1.0).astype(np.float32)
    return reg_day


def build_enriched_basin_panel(
    climate_parquet: Path | str,
    cnes_area_parquet: Path | str,
    df_reg: pd.DataFrame,
    df_sedec: pd.DataFrame,
    df_wind: Optional[pd.DataFrame] = None,
    df_ana: Optional[pd.DataFrame] = None,
    df_infra_daily: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Constrói painel diário regional com roteamento causal e alvo ponderado por população."""
    df_c = pd.read_parquet(climate_parquet)
    df_c["id_municipio"] = df_c["id_municipio"].astype(str).str.strip()
    df_c = df_c[df_c["id_municipio"].str.startswith("43")].copy()
    df_c = df_c.merge(df_reg[["id_municipio", "regiao_hidro"]], on="id_municipio", how="inner")
    df_c["data"] = pd.to_datetime(df_c["data"]).dt.normalize()

    for col in ["latitude", "longitude"]:
        if col not in df_c.columns:
            df_c[col] = 0.0

    # Detecta automaticamente se há colunas de vento no parquet climático
    parquet_wind_cols = [
        c
        for c in df_c.columns
        if "vento" in c.lower() and pd.api.types.is_numeric_dtype(df_c[c]) and c not in WIND_DYNAMIC_COLS
    ]
    agg_dict = {
        "precip_media_mm": ("precip_dia_mm", "mean"),
        "precip_max_reg_mm": ("precip_dia_mm", "max"),
        "precip_max_horaria_mm": ("precip_max_horaria_mm", "max"),
        "temp_media_c": ("temp_media_c", "mean"),
        "temp_max_c": ("temp_max_c", "max"),
        "umidade_rel_media": ("umidade_rel_media", "mean"),
        "pressao_media_hpa": ("pressao_media_hpa", "mean"),
        "indice_impermeabilidade": ("indice_impermeabilidade", "mean"),
        "indice_retencao_vegetal": ("indice_retencao_vegetal", "mean"),
        "altitude_estacao": ("altitude_estacao", "mean"),
        "latitude": ("latitude", "mean"),
        "longitude": ("longitude", "mean"),
    }
    for wc in parquet_wind_cols:
        agg_dict[wc] = (wc, "mean")

    reg_day = df_c.groupby(["regiao_hidro", "data"], as_index=False).agg(**agg_dict)

    all_dates = pd.date_range(reg_day["data"].min(), reg_day["data"].max(), freq="D")
    regioes = sorted(reg_day["regiao_hidro"].unique())
    full_idx = pd.MultiIndex.from_product([regioes, all_dates], names=["regiao_hidro", "data"])
    reg_day = reg_day.set_index(["regiao_hidro", "data"]).reindex(full_idx).reset_index()
    reg_day["ano"] = reg_day["data"].dt.year
    reg_day["mes"] = reg_day["data"].dt.month

    for col in ["precip_media_mm", "precip_max_reg_mm", "precip_max_horaria_mm"]:
        reg_day[col] = reg_day[col].fillna(0.0)
    fill_cols = [
        "temp_media_c",
        "temp_max_c",
        "umidade_rel_media",
        "pressao_media_hpa",
        "indice_impermeabilidade",
        "indice_retencao_vegetal",
        "altitude_estacao",
        "latitude",
        "longitude",
    ] + parquet_wind_cols
    for col in fill_cols:
        reg_day[col] = reg_day.groupby("regiao_hidro")[col].ffill().bfill()
        reg_day[col] = reg_day[col].fillna(reg_day[col].median())

    reg_day = reg_day.sort_values(["regiao_hidro", "data"]).reset_index(drop=True)
    doy = reg_day["data"].dt.dayofyear.to_numpy(dtype=np.float32)
    reg_day["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    reg_day["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    # Saturação do solo (API) + janelas até 30 dias + queda barométrica de 3 dias
    k_decay = 0.88
    p_arr = reg_day["precip_media_mm"].to_numpy(dtype=np.float32)
    r_arr = reg_day["regiao_hidro"].to_numpy()
    api = np.zeros(len(reg_day), dtype=np.float32)
    for t in range(1, len(p_arr)):
        if r_arr[t] == r_arr[t - 1]:
            api[t] = k_decay * api[t - 1] + p_arr[t - 1]
    reg_day["api_indice_saturacao"] = api

    grp = reg_day.groupby("regiao_hidro")
    for w in [3, 7, 15, 30]:
        reg_day[f"precip_acum_{w}d"] = grp["precip_media_mm"].transform(lambda s: s.rolling(w, min_periods=1).sum())

    # Queda barométrica em 3 dias (proxy de entrada de frente fria / vento Sul)
    reg_day["pressao_var_3d"] = grp["pressao_media_hpa"].transform(lambda s: s - s.shift(3)).fillna(0.0)

    sat_norm = np.tanh(reg_day["api_indice_saturacao"] / 120.0)
    imp = reg_day["indice_impermeabilidade"].clip(0.0, 1.0)
    reg_day["escoamento_efetivo_mm"] = reg_day["precip_media_mm"] * (imp + (1.0 - imp) * sat_norm)
    reg_day["escoamento_acum_7d"] = grp["escoamento_efetivo_mm"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    reg_day["escoamento_acum_15d"] = grp["escoamento_efetivo_mm"].transform(lambda s: s.rolling(15, min_periods=1).sum())

    # Roteamento causal Montante -> Jusante (7d, 15d e 30d)
    pivs = {
        "c7": reg_day.pivot(index="data", columns="regiao_hidro", values="precip_acum_7d").shift(1).fillna(0.0),
        "c15": reg_day.pivot(index="data", columns="regiao_hidro", values="precip_acum_15d").shift(1).fillna(0.0),
        "c30": reg_day.pivot(index="data", columns="regiao_hidro", values="precip_acum_30d").shift(1).fillna(0.0),
        "e7": reg_day.pivot(index="data", columns="regiao_hidro", values="escoamento_acum_7d").shift(1).fillna(0.0),
        "e15": reg_day.pivot(index="data", columns="regiao_hidro", values="escoamento_acum_15d").shift(1).fillna(0.0),
    }
    up_maps = {k: {} for k in pivs}
    for reg in regioes:
        ups = [u for u in UPSTREAM_BASINS.get(reg, []) if u in pivs["c7"].columns]
        for k, pv in pivs.items():
            up_maps[k][reg] = pv[ups].mean(axis=1) if ups else pd.Series(0.0, index=pv.index)

    up_df = pd.concat(
        {
            "montante_chuva_7d": pd.DataFrame(up_maps["c7"]).stack(),
            "montante_chuva_15d": pd.DataFrame(up_maps["c15"]).stack(),
            "montante_chuva_30d": pd.DataFrame(up_maps["c30"]).stack(),
            "montante_escoamento_7d": pd.DataFrame(up_maps["e7"]).stack(),
            "montante_escoamento_15d": pd.DataFrame(up_maps["e15"]).stack(),
        },
        axis=1,
    ).reset_index()
    up_df.columns = [
        "data",
        "regiao_hidro",
        "montante_chuva_7d",
        "montante_chuva_15d",
        "montante_chuva_30d",
        "montante_escoamento_7d",
        "montante_escoamento_15d",
    ]
    reg_day = reg_day.merge(up_df, on=["regiao_hidro", "data"], how="left")

    dyn_cols = list(BASE_DYNAMIC_COLS) + parquet_wind_cols
    if parquet_wind_cols:
        print(f"[CLIMA] Colunas de vento do Parquet incorporadas: {parquet_wind_cols}")

    if df_wind is not None and not df_wind.empty:
        reg_day = _merge_wind_features(reg_day, df_wind, df_reg)
        dyn_cols.extend(WIND_DYNAMIC_COLS)
        print(f"[VENTO BARRA RG] {len(WIND_DYNAMIC_COLS)} variáveis vetoriais integradas ao painel.")

    if df_ana is not None and not df_ana.empty:
        reg_day = _merge_ana_features(reg_day, df_ana, regioes)
        dyn_cols.extend(ANA_DYNAMIC_COLS)
        print(f"[ANA] {len(ANA_DYNAMIC_COLS)} variáveis fluviométricas integradas ao painel.")

    if df_infra_daily is not None and not df_infra_daily.empty:
        reg_day = _merge_infra_features(reg_day, df_infra_daily)
        dyn_cols.extend(INFRA_BINARY_COLS)
        print(f"[INFRA] Indicadores binários integrados: {INFRA_BINARY_COLS}")

    # Alvo ponderado por população (elimina viés de municípios rurais vs. metrópoles)
    df_pop = pd.read_parquet(cnes_area_parquet)
    df_pop["id_municipio"] = df_pop["id_municipio"].astype(str).str.strip()
    pop_mun = (
        df_pop[df_pop["id_municipio"].str.startswith("43")]
        .sort_values(["ano", "mes"])
        .groupby("id_municipio", as_index=False)["populacao"]
        .last()
    )
    df_reg_pop = df_reg[["id_municipio", "regiao_hidro"]].merge(pop_mun, on="id_municipio", how="left")
    df_reg_pop["populacao"] = df_reg_pop["populacao"].fillna(df_reg_pop["populacao"].median())
    pop_tot_reg = df_reg_pop.groupby("regiao_hidro")["populacao"].sum().to_dict()

    sedec_reg = df_sedec.merge(df_reg_pop, on="id_municipio", how="inner")
    expanded = []
    for offset in range(-9, 1):
        tmp = sedec_reg[["id_municipio", "regiao_hidro", "populacao", "data_decreto"]].copy()
        tmp["data"] = tmp["data_decreto"] + pd.Timedelta(days=offset)
        expanded.append(tmp[["id_municipio", "regiao_hidro", "populacao", "data"]])

    sedec_daily = pd.concat(expanded, ignore_index=True).drop_duplicates(["id_municipio", "data"])
    sedec_agg = sedec_daily.groupby(["regiao_hidro", "data"], as_index=False).agg(
        mun_atingidos_real=("id_municipio", "nunique"),
        pop_atingida_real=("populacao", "sum"),
    )

    reg_day = reg_day.merge(sedec_agg, on=["regiao_hidro", "data"], how="left")
    reg_day["mun_atingidos_real"] = reg_day["mun_atingidos_real"].fillna(0).astype(int)
    reg_day["pop_atingida_real"] = reg_day["pop_atingida_real"].fillna(0.0)
    reg_day["pop_total_regiao"] = reg_day["regiao_hidro"].map(pop_tot_reg)
    reg_day["pct_pop_atingida_real"] = (
        (reg_day["pop_atingida_real"] / reg_day["pop_total_regiao"].clip(lower=1.0)) * 100.0
    ).clip(0.0, 100.0)
    reg_day["alvo_enchente_reg"] = (reg_day["mun_atingidos_real"] >= 1).astype(np.int64)

    return reg_day.sort_values(["regiao_hidro", "data"]).reset_index(drop=True), dyn_cols


class RegionalBasinDataset(Dataset):
    """Dataset 2D indexado que fatia janelas temporais por região hidrográfica."""

    def __init__(self, df: pd.DataFrame, dyn_cols: List[str], reg_to_idx: Dict[str, int], seq_len: int = 21):
        self.seq_len = seq_len
        self.dyn_2d = df[dyn_cols].to_numpy(dtype=np.float32)
        self.stat_2d = df[REG_STATIC_COLS].to_numpy(dtype=np.float32)
        self.reg_1d = df["regiao_hidro"].map(reg_to_idx).fillna(0).to_numpy(dtype=np.int64)
        self.ycls_1d = df["alvo_enchente_reg"].to_numpy(dtype=np.int64)
        self.ysev_1d = (df["pct_pop_atingida_real"].to_numpy(dtype=np.float32)) / 10.0

        regs = df["regiao_hidro"].to_numpy()
        valid_end = []
        start, n = 0, len(regs)
        while start < n:
            end = start + 1
            while end < n and regs[end] == regs[start]:
                end += 1
            if end - start >= seq_len:
                valid_end.append(np.arange(start + seq_len - 1, end, dtype=np.int64))
            start = end

        self.end_indices = np.concatenate(valid_end)
        meta_cols = [
            "regiao_hidro",
            "data",
            "ano",
            "mes",
            "alvo_enchente_reg",
            "pct_pop_atingida_real",
            "temp_media_c",
        ] + [c for c in INFRA_BINARY_COLS if c in df.columns]
        self.meta_df = df.iloc[self.end_indices][meta_cols].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, idx: int):
        e = self.end_indices[idx]
        return (
            torch.from_numpy(self.dyn_2d[e - self.seq_len + 1 : e + 1]),
            torch.from_numpy(self.stat_2d[e]),
            torch.tensor(self.reg_1d[e]),
            torch.tensor(self.ycls_1d[e]),
            torch.tensor(self.ysev_1d[e]),
        )


class HydroFiLMRegionalBasinNet(nn.Module):
    """Rede Bi-GRU + FiLM + Multi-Head Attention + Ramo Monotônico Estrutural (Solo e Infra)."""

    def __init__(
        self,
        num_dyn: int,
        rain_3d_idx: int,
        num_regions: int,
        hidden_dim: int = 48,
        dropout: float = 0.25,
        dike_idx: Optional[int] = None,
        pump_idx: Optional[int] = None,
    ):
        super().__init__()
        self.rain_3d_idx = rain_3d_idx
        self.dike_idx = dike_idx
        self.pump_idx = pump_idx
        self.gru = nn.GRU(
            input_size=num_dyn,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        seq_dim = hidden_dim * 2
        self.reg_emb = nn.Embedding(num_regions + 1, 4)
        self.film_gen = nn.Sequential(
            nn.Linear(4 + 3, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, seq_dim * 2),
        )
        self.attn = nn.MultiheadAttention(seq_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(seq_dim)
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(seq_dim, 1)
        self.sev_head = nn.Sequential(nn.Linear(seq_dim, 1), nn.Softplus())

        # Ramo Monotônico Estrutural (solo + infraestrutura)
        self.raw_w_imp = nn.Parameter(torch.tensor(-1.5))
        self.raw_w_veg = nn.Parameter(torch.tensor(-1.5))
        self.raw_w_int = nn.Parameter(torch.tensor(-2.0))
        self.raw_w_dike = nn.Parameter(torch.tensor(-2.0))
        self.raw_w_pump = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x_dyn, x_stat, reg_idx):
        h_seq, _ = self.gru(x_dyn)
        geo = torch.cat([self.reg_emb(reg_idx), x_stat[:, 2:5]], dim=-1)
        gamma, beta = torch.chunk(self.film_gen(geo), 2, dim=-1)
        h_mod = (1.0 + gamma.unsqueeze(1)) * h_seq + beta.unsqueeze(1)
        attn_out, _ = self.attn(h_mod, h_mod, h_mod)
        h_last = self.norm(h_mod + self.dropout(attn_out))[:, -1, :]

        w_imp = F.softplus(self.raw_w_imp)
        w_veg = F.softplus(self.raw_w_veg)
        w_int = F.softplus(self.raw_w_int)
        rain_gate = F.softplus(x_dyn[:, -1, self.rain_3d_idx])
        soil_eff = (w_imp + w_int * rain_gate) * x_stat[:, 0] - w_veg * x_stat[:, 1]

        infra_eff = torch.zeros_like(soil_eff)
        if self.dike_idx is not None:
            infra_eff = infra_eff + F.softplus(self.raw_w_dike) * x_dyn[:, -1, self.dike_idx]
        if self.pump_idx is not None:
            infra_eff = infra_eff + F.softplus(self.raw_w_pump) * x_dyn[:, -1, self.pump_idx]

        return self.cls_head(h_last).squeeze(-1) + soil_eff + infra_eff, self.sev_head(h_last).squeeze(-1) + infra_eff


def train_and_predict_basin_2024(
    df_panel: pd.DataFrame,
    dyn_cols: List[str],
    epochs: int = 10,
    seq_len: int = 21,
    device: Optional[str] = None,
    model_save_path: Optional[Path | str] = None,
    model_load_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Treina (ou carrega checkpoint) a HydroFiLMRegionalBasinNet e avalia out-of-time em 2024."""
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cont_dyn_cols = [c for c in dyn_cols if c not in INFRA_BINARY_COLS]
    load_path = Path(model_load_path) if model_load_path is not None else None

    if load_path is not None:
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint do modelo não encontrado em: {load_path}")
        ckpt = torch.load(load_path, map_location=resolved_device, weights_only=False)
        if ckpt["dyn_cols"] != dyn_cols:
            raise ValueError(
                "As colunas dinâmicas do checkpoint diferem das atuais "
                "(verifique as flags --no-wind / --no-ana / --no-infra)."
            )
        reg_to_idx = ckpt["reg_to_idx"]
        seq_len = int(ckpt["seq_len"])
        dyn_m, dyn_s = ckpt["dyn_m"], ckpt["dyn_s"]
        stat_m, stat_s = ckpt["stat_m"], ckpt["stat_s"]
        dike_idx, pump_idx = ckpt["dike_idx"], ckpt["pump_idx"]
    else:
        unique_regs = sorted(df_panel["regiao_hidro"].unique())
        reg_to_idx = {r: i + 1 for i, r in enumerate(unique_regs)}
        df_tr_raw = df_panel[df_panel["ano"] <= 2023]
        if cont_dyn_cols:
            dyn_m = df_tr_raw[cont_dyn_cols].mean().to_numpy()
            raw_std = df_tr_raw[cont_dyn_cols].std().to_numpy()
            dyn_s = np.where(raw_std >= 1e-6, raw_std, 1.0)
        else:
            dyn_m = np.array([], dtype=np.float32)
            dyn_s = np.array([], dtype=np.float32)
        stat_m = df_tr_raw[REG_STATIC_COLS].mean().to_numpy()
        raw_stat_std = df_tr_raw[REG_STATIC_COLS].std().to_numpy()
        stat_s = np.where(raw_stat_std >= 1e-6, raw_stat_std, 1.0)
        dike_idx = dyn_cols.index("falha_dique") if "falha_dique" in dyn_cols else None
        pump_idx = dyn_cols.index("falha_casa_bombas") if "falha_casa_bombas" in dyn_cols else None

    # Normalização estrita em 2010–2023 (Zero Leakage; nunca aplica z-score em binárias de infra)
    df_norm = df_panel.copy()
    if cont_dyn_cols:
        df_norm[cont_dyn_cols] = (df_panel[cont_dyn_cols] - dyn_m) / dyn_s
    df_norm[REG_STATIC_COLS] = (df_panel[REG_STATIC_COLS] - stat_m) / stat_s

    # Janela de 21 dias: inicia em 11/12/2023 para prever desde 01/01/2024
    test_24_ds = RegionalBasinDataset(
        df_norm[(df_norm["data"] >= "2023-12-11") & (df_norm["ano"] <= 2024)],
        dyn_cols,
        reg_to_idx,
        seq_len=seq_len,
    )
    test_loader = DataLoader(test_24_ds, batch_size=512, shuffle=False, num_workers=2)

    model = HydroFiLMRegionalBasinNet(
        num_dyn=len(dyn_cols),
        rain_3d_idx=dyn_cols.index("precip_acum_3d"),
        num_regions=len(reg_to_idx),
        hidden_dim=48,
        dropout=0.25,
        dike_idx=dike_idx,
        pump_idx=pump_idx,
    ).to(resolved_device)

    if load_path is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[MODELO] Pesos carregados de: {load_path} (treino ignorado).")
    else:
        train_ds = RegionalBasinDataset(df_norm[df_norm["ano"] <= 2023], dyn_cols, reg_to_idx, seq_len=seq_len)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, epochs=epochs, steps_per_epoch=len(train_loader))
        huber = nn.HuberLoss(delta=1.0)

        print(f"\n--- Treinando HydroFiLMRegionalBasinNet ({len(dyn_cols)} features dinâmicas | seq_len={seq_len}) ---")
        for ep in range(1, epochs + 1):
            model.train()
            ep_loss = 0.0
            for x_dyn, x_stat, reg_idx, y_cls, y_sev in train_loader:
                opt.zero_grad()
                logits, pred_sev = model(
                    x_dyn.to(resolved_device),
                    x_stat.to(resolved_device),
                    reg_idx.to(resolved_device),
                )
                y_c = y_cls.to(resolved_device).float()
                bce = F.binary_cross_entropy_with_logits(logits, y_c, reduction="none")
                p_t = torch.sigmoid(logits) * y_c + (1 - torch.sigmoid(logits)) * (1 - y_c)
                focal = ((0.75 * y_c + 0.25 * (1 - y_c)) * ((1 - p_t) ** 2) * bce).mean()
                loss = focal + 0.50 * huber(pred_sev, y_sev.to(resolved_device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                ep_loss += loss.item()
            if ep in (1, 5, epochs):
                print(f"Epoch {ep:02d}/{epochs:02d} | Loss: {ep_loss / len(train_loader):.4f}")

        if model_save_path is not None:
            save_path = Path(model_save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dyn_cols": dyn_cols,
                    "reg_to_idx": reg_to_idx,
                    "seq_len": seq_len,
                    "dyn_m": dyn_m,
                    "dyn_s": dyn_s,
                    "stat_m": stat_m,
                    "stat_s": stat_s,
                    "dike_idx": dike_idx,
                    "pump_idx": pump_idx,
                },
                save_path,
            )
            print(f"[MODELO] Pesos salvos em: {save_path}")

    # Previsão Out-of-Time 2024 (% População Atingida)
    model.eval()
    p_list, sev_list = [], []
    with torch.no_grad():
        for x_dyn, x_stat, reg_idx, _, _ in test_loader:
            logits, pred_sev = model(
                x_dyn.to(resolved_device),
                x_stat.to(resolved_device),
                reg_idx.to(resolved_device),
            )
            p_list.append(torch.sigmoid(logits).cpu().numpy())
            sev_list.append(np.clip(pred_sev.cpu().numpy() * 10.0, 0.0, 100.0))

    df_pred_24 = test_24_ds.meta_df.copy()
    df_pred_24["prob_enchente_pred"] = np.concatenate(p_list)
    df_pred_24["pct_pop_atingida_pred"] = np.concatenate(sev_list)
    df_pred_24 = df_pred_24[df_pred_24["ano"] == 2024].copy()

    print("\n=== AUDITORIA DE ALAGAMENTO 2024 (% POPULAÇÃO ATINGIDA POR REGIÃO) ===")
    for reg, sub in df_pred_24.groupby("regiao_hidro"):
        mae_pop = mean_absolute_error(sub["pct_pop_atingida_real"], sub["pct_pop_atingida_pred"])
        if len(sub["alvo_enchente_reg"].unique()) > 1:
            roc = roc_auc_score(sub["alvo_enchente_reg"], sub["prob_enchente_pred"])
            pr = average_precision_score(sub["alvo_enchente_reg"], sub["prob_enchente_pred"])
            brier = brier_score_loss(sub["alvo_enchente_reg"], sub["prob_enchente_pred"])
            print(f"{reg:36s} | ROC: {roc:.4f} | PR: {pr:.4f} | Brier: {brier:.4f} | MAE (% Pop): {mae_pop:5.2f}%")

    return df_pred_24
