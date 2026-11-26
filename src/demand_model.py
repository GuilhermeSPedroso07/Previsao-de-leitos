"""Estágio 2: Modelagem de demanda real de leitos (SIH/SUS vs. CNES-SUS) por coorte."""

import calendar
from pathlib import Path
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Grafo de adjacência rodoviária/referência SUS (transbordamento de pacientes)
NEIGHBOR_REGIONS: Dict[str, List[str]] = {
    "Metropolitana e Bacia do Guaíba": [
        "Serra Gaúcha (Alto Taquari/Caí)",
        "Vales (Taquari e Rio Pardo)",
        "Região Sul e Lagoa dos Patos",
    ],
    "Serra Gaúcha (Alto Taquari/Caí)": [
        "Metropolitana e Bacia do Guaíba",
        "Planalto Norte (Alto Jacuí/Uruguai)",
        "Vales (Taquari e Rio Pardo)",
    ],
    "Vales (Taquari e Rio Pardo)": [
        "Metropolitana e Bacia do Guaíba",
        "Serra Gaúcha (Alto Taquari/Caí)",
        "Planalto Norte (Alto Jacuí/Uruguai)",
        "Depressão Central (Médio Jacuí)",
        "Região Sul e Lagoa dos Patos",
    ],
    "Planalto Norte (Alto Jacuí/Uruguai)": [
        "Serra Gaúcha (Alto Taquari/Caí)",
        "Vales (Taquari e Rio Pardo)",
        "Depressão Central (Médio Jacuí)",
        "Noroeste e Missões",
    ],
    "Depressão Central (Médio Jacuí)": [
        "Vales (Taquari e Rio Pardo)",
        "Planalto Norte (Alto Jacuí/Uruguai)",
        "Noroeste e Missões",
        "Fronteira Oeste e Campanha",
        "Região Sul e Lagoa dos Patos",
    ],
    "Noroeste e Missões": [
        "Planalto Norte (Alto Jacuí/Uruguai)",
        "Depressão Central (Médio Jacuí)",
        "Fronteira Oeste e Campanha",
    ],
    "Fronteira Oeste e Campanha": [
        "Noroeste e Missões",
        "Depressão Central (Médio Jacuí)",
        "Região Sul e Lagoa dos Patos",
    ],
    "Região Sul e Lagoa dos Patos": [
        "Metropolitana e Bacia do Guaíba",
        "Vales (Taquari e Rio Pardo)",
        "Depressão Central (Médio Jacuí)",
        "Fronteira Oeste e Campanha",
    ],
}

STAGE2_FEATURES: List[str] = [
    "sev_log",
    "disrupcao_extrema",
    "sev_lag_1m",
    "sev_lag_2m",
    "sev_vizinhos_max",
    "temp_frio_idx",
    "mes_sin",
    "mes_cos",
]

INFRA_STAGE2_FEATURES: List[str] = [
    "falha_dique_mes",
    "falha_casa_bombas_mes",
    "dias_falha_infra_mes",
]


def _slugify_title(title: str) -> str:
    """Converte o título da coorte em nome de arquivo seguro para salvar gráficos."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()).strip("_")
    return f"{slug}.png"


def build_stage2_monthly_features(df_src: pd.DataFrame, sev_col: str) -> pd.DataFrame:
    """Constrói os parâmetros mensais do Estágio 2 (disrupção, rebote, vizinhança, frio e infra)."""
    df_work = df_src.copy()
    has_infra = "falha_dique" in df_work.columns and "falha_casa_bombas" in df_work.columns
    agg_kwargs = {
        "sev_max": (sev_col, "max"),
        "sev_med": (sev_col, "mean"),
        "temp_media_mes": ("temp_media_c", "mean"),
    }
    if has_infra:
        df_work["falha_infra_ativa"] = np.maximum(
            df_work["falha_dique"].astype(float),
            df_work["falha_casa_bombas"].astype(float),
        )
        agg_kwargs["falha_dique_mes"] = ("falha_dique", "max")
        agg_kwargs["falha_casa_bombas_mes"] = ("falha_casa_bombas", "max")
        agg_kwargs["dias_falha_infra_mes"] = ("falha_infra_ativa", "sum")

    m = (
        df_work.groupby(["regiao_hidro", "ano", "mes"], as_index=False)
        .agg(**agg_kwargs)
        .sort_values(["regiao_hidro", "ano", "mes"])
    )

    m["sev_log"] = np.log1p(m["sev_max"])
    m["disrupcao_extrema"] = np.log1p(np.maximum(0.0, m["sev_max"] - 25.0))
    m["sev_lag_1m"] = np.log1p(m.groupby("regiao_hidro")["sev_max"].shift(1).fillna(0.0))
    m["sev_lag_2m"] = np.log1p(m.groupby("regiao_hidro")["sev_max"].shift(2).fillna(0.0))
    m["temp_frio_idx"] = np.maximum(0.0, 20.0 - m["temp_media_mes"])
    m["mes_sin"] = np.sin(2.0 * np.pi * m["mes"] / 12.0)
    m["mes_cos"] = np.cos(2.0 * np.pi * m["mes"] / 12.0)

    # Transbordamento hospitalar das regiões vizinhas (no mesmo mês)
    piv_sev = m.pivot(index=["ano", "mes"], columns="regiao_hidro", values="sev_max").fillna(0.0)
    viz_dict = {}
    for reg in piv_sev.columns:
        viz = [v for v in NEIGHBOR_REGIONS.get(reg, []) if v in piv_sev.columns]
        viz_dict[reg] = np.log1p(piv_sev[viz].max(axis=1)) if viz else pd.Series(0.0, index=piv_sev.index)

    viz_df = pd.DataFrame(viz_dict).stack().reset_index()
    viz_df.columns = ["ano", "mes", "regiao_hidro", "sev_vizinhos_max"]
    return m.merge(viz_df, on=["ano", "mes", "regiao_hidro"], how="left")


def evaluate_separated_demands(
    df_panel: pd.DataFrame,
    df_pred_24: pd.DataFrame,
    df_reg: pd.DataFrame,
    df_sih: pd.DataFrame,
    df_cnes_sus: pd.DataFrame,
    occupancy_target: float = 0.85,
    save_plots_dir: Optional[Path | str] = None,
    show_plots: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Treina Ridge em 2011–2019 e 2023 e avalia a demanda real de leitos SUS em 2024 por coorte."""
    df_reg_6 = df_reg[["id_municipio", "regiao_hidro"]].copy()
    df_reg_6["id_mun_6"] = df_reg_6["id_municipio"].astype(str).str.strip().str[:6]
    df_reg_6 = df_reg_6[["id_mun_6", "regiao_hidro"]].drop_duplicates("id_mun_6")

    # População por região (já calculada no df_panel) para o modo aditivo por 100k hab.
    pop_reg = df_panel.groupby("regiao_hidro", as_index=False)["pop_total_regiao"].first()

    sih_reg = (
        df_sih.merge(df_reg_6, on="id_mun_6", how="inner")
        .groupby(["regiao_hidro", "ano", "mes"], as_index=False)
        .agg(
            diarias_urg_uti=("diarias_urg_uti", "sum"),
            diarias_urg_total=("diarias_urg_total", "sum"),
            diarias_cid_uti=("diarias_cid_uti", "sum"),
            diarias_cid_total=("diarias_cid_total", "sum"),
        )
    )
    sih_reg = sih_reg.merge(pop_reg, on="regiao_hidro", how="left")
    sih_reg["dias_mes"] = [
        calendar.monthrange(int(a), int(m))[1]
        for a, m in zip(sih_reg["ano"], sih_reg["mes"])
    ]

    cnes_24 = (
        df_cnes_sus[df_cnes_sus["ano"] == 2024]
        .merge(df_reg_6, on="id_mun_6", how="inner")
        .groupby(["regiao_hidro", "mes"], as_index=False)
        .agg(oferta_total_sus=("oferta_total_sus", "sum"), oferta_uti_sus=("oferta_uti_sus", "sum"))
    )

    extra_infra_cols = [
        c for c in ("falha_dique", "falha_casa_bombas") if c in df_panel.columns and c in df_pred_24.columns
    ]

    hist_feat = build_stage2_monthly_features(df_panel[df_panel["ano"] <= 2023], "pct_pop_atingida_real")
    nov_dez23 = df_panel[(df_panel["ano"] == 2023) & (df_panel["mes"].isin([11, 12]))][
        ["regiao_hidro", "ano", "mes", "pct_pop_atingida_real", "temp_media_c"] + extra_infra_cols
    ].copy()

    feat_24_real = build_stage2_monthly_features(
        pd.concat([
            nov_dez23,
            df_pred_24[["regiao_hidro", "ano", "mes", "pct_pop_atingida_real", "temp_media_c"] + extra_infra_cols],
        ]),
        "pct_pop_atingida_real",
    )
    feat_24_pred = build_stage2_monthly_features(
        pd.concat([
            nov_dez23.rename(columns={"pct_pop_atingida_real": "pct_pop_atingida_pred"}),
            df_pred_24[["regiao_hidro", "ano", "mes", "pct_pop_atingida_pred", "temp_media_c"] + extra_infra_cols],
        ]),
        "pct_pop_atingida_pred",
    )
    feat_24_real = feat_24_real[feat_24_real["ano"] == 2024]
    feat_24_pred = feat_24_pred[feat_24_pred["ano"] == 2024]

    # (título, coluna de diárias, coluna CNES, usar_modo_aditivo_100k)
    cohorts = [
        ("URGÊNCIA — UTI SUS", "diarias_urg_uti", "oferta_uti_sus", False),
        ("URGÊNCIA — TOTAL SUS", "diarias_urg_total", "oferta_total_sus", False),
        ("CIDs SENSÍVEIS A ENCHENTES — TOTAL SUS", "diarias_cid_total", "oferta_total_sus", True),
        ("CIDs SENSÍVEIS A ENCHENTES — UTI SUS", "diarias_cid_uti", "oferta_uti_sus", True),
    ]

    plots_dir = Path(save_plots_dir) if save_plots_dir is not None else None
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, pd.DataFrame] = {}
    for title, diaria_col, cnes_col, additive_mode in cohorts:
        sub = sih_reg.copy()
        sub["leitos_demanda_real"] = sub[diaria_col] / (sub["dias_mes"] * occupancy_target)

        if additive_mode:
            # MODO ADITIVO (por 100k hab., ancorado na média do ano anterior -> nunca trava em 0)
            sub["taxa_100k"] = (sub["leitos_demanda_real"] / sub["pop_total_regiao"].clip(lower=1.0)) * 100_000.0
            media_ant = sub.groupby(["regiao_hidro", "ano"], as_index=False)["taxa_100k"].mean()
            media_ant["ano"] = media_ant["ano"] + 1
            media_ant = media_ant.rename(columns={"taxa_100k": "base_taxa_100k"})
            sub = sub.merge(media_ant, on=["regiao_hidro", "ano"], how="inner")
            sub["demanda_base"] = (sub["base_taxa_100k"] * sub["pop_total_regiao"]) / 100_000.0
            target_col = "delta_100k"
            sub[target_col] = sub["taxa_100k"] - sub["base_taxa_100k"]
        else:
            # MODO MULTIPLICATIVO (para séries densas de Urgência)
            dez_ant = sub[sub["mes"] == 12][["regiao_hidro", "ano", "leitos_demanda_real"]].copy()
            dez_ant["ano"] = dez_ant["ano"] + 1
            dez_ant = dez_ant.rename(columns={"leitos_demanda_real": "demanda_base"})
            sub = sub.merge(dez_ant, on=["regiao_hidro", "ano"], how="inner")
            target_col = "pct_surto"
            sub[target_col] = (sub["leitos_demanda_real"] - sub["demanda_base"]) / sub["demanda_base"].clip(lower=1.0)

        df_tr = sub[sub["ano"].isin([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2023])].merge(
            hist_feat, on=["regiao_hidro", "ano", "mes"], how="inner"
        )
        base_24 = sub[sub["ano"] == 2024].copy()
        cmp_real = base_24.merge(feat_24_real, on=["regiao_hidro", "ano", "mes"], how="inner")
        cmp_pred = base_24.merge(feat_24_pred, on=["regiao_hidro", "ano", "mes"], how="inner")

        active_feats = list(STAGE2_FEATURES) + [
            c
            for c in INFRA_STAGE2_FEATURES
            if c in df_tr.columns and c in cmp_real.columns and c in cmp_pred.columns and df_tr[c].std() > 1e-6
        ]

        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        model.fit(df_tr[active_feats], df_tr[target_col])

        df_out = cmp_real[["regiao_hidro", "mes", "demanda_base", "leitos_demanda_real"]].copy()
        if additive_mode:
            pop_24 = cmp_real["pop_total_regiao"].to_numpy()
            base_100k = cmp_real["base_taxa_100k"].to_numpy()
            df_out["demanda_via_alag_pred"] = np.maximum(
                0.0, (base_100k + model.predict(cmp_pred[active_feats])) * pop_24 / 100_000.0
            )
            df_out["demanda_via_alag_real"] = np.maximum(
                0.0, (base_100k + model.predict(cmp_real[active_feats])) * pop_24 / 100_000.0
            )
        else:
            df_out["demanda_via_alag_pred"] = np.maximum(
                0.0, df_out["demanda_base"] * (1.0 + model.predict(cmp_pred[active_feats]))
            )
            df_out["demanda_via_alag_real"] = np.maximum(
                0.0, df_out["demanda_base"] * (1.0 + model.predict(cmp_real[active_feats]))
            )

        df_out = df_out.merge(cnes_24[["regiao_hidro", "mes", cnes_col]], on=["regiao_hidro", "mes"], how="left")
        df_out = df_out.rename(columns={cnes_col: "oferta_sus_cnes"})

        print(f"\n=== {title} ({'ADITIVO/100k' if additive_mode else 'MULTIPLICATIVO'}) ===")
        print(
            f"2024 Via Alag. PREVISTO -> MAE: {mean_absolute_error(df_out['leitos_demanda_real'], df_out['demanda_via_alag_pred']):6.2f} "
            f"| R²: {r2_score(df_out['leitos_demanda_real'], df_out['demanda_via_alag_pred']):.4f}"
        )
        print(
            f"2024 Via Alag. REAL     -> MAE: {mean_absolute_error(df_out['leitos_demanda_real'], df_out['demanda_via_alag_real']):6.2f} "
            f"| R²: {r2_score(df_out['leitos_demanda_real'], df_out['demanda_via_alag_real']):.4f}"
        )

        if plots_dir is not None or show_plots:
            regioes = sorted(df_out["regiao_hidro"].unique())
            fig, axes = plt.subplots(4, 2, figsize=(14, 11.5), sharex=True)
            axes_flat = axes.flatten()
            show_cnes = "URGÊNCIA" in title
            for i, reg in enumerate(regioes):
                sub_r = df_out[df_out["regiao_hidro"] == reg].sort_values("mes")
                if show_cnes:
                    axes_flat[i].plot(
                        sub_r["mes"], sub_r["oferta_sus_cnes"], "k:", linewidth=1.6, label="Oferta Total SUS (CNES)"
                    )
                axes_flat[i].plot(
                    sub_r["mes"],
                    sub_r["leitos_demanda_real"],
                    "o-",
                    color="#1f77b4",
                    linewidth=2,
                    label="Demanda Real (SIH 85%)",
                )
                axes_flat[i].plot(
                    sub_r["mes"],
                    sub_r["demanda_via_alag_real"],
                    "^--",
                    color="#2ca02c",
                    linewidth=1.6,
                    label="Prev. via Alag. Real",
                )
                axes_flat[i].plot(
                    sub_r["mes"],
                    sub_r["demanda_via_alag_pred"],
                    "s--",
                    color="#ff7f0e",
                    linewidth=1.6,
                    label="Prev. via Alag. Previsto",
                )
                axes_flat[i].axvline(5, color="red", linestyle=":", alpha=0.45)
                axes_flat[i].axvline(7, color="purple", linestyle="--", alpha=0.35)
                axes_flat[i].set_title(reg, fontsize=10, fontweight="bold")
                axes_flat[i].grid(alpha=0.3)
                if i == 0:
                    axes_flat[i].legend(fontsize=7.2)
            plt.suptitle(f"RS 2024 — {title}: Real vs. Previsto", y=1.01)
            plt.tight_layout()
            if plots_dir is not None:
                fig.savefig(plots_dir / _slugify_title(title), dpi=150, bbox_inches="tight")
            if show_plots:
                plt.show()
            else:
                plt.close(fig)

        results[title] = df_out

    return results
