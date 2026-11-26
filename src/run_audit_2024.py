"""Orquestrador CLI da auditoria RS 2024/5ª Fase (Bacias + Vento Barra RG + ANA + Infra + SIH/SUS)."""

import argparse
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple
import warnings

import pandas as pd
import torch

# Garante importação absoluta ao executar diretamente via `python src/run_audit_2024.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ana_telemetry import fetch_ana_rs_daily_cotas, fetch_realtime_rs_cotas
from src.basin_model import build_enriched_basin_panel, train_and_predict_basin_2024
from src.demand_model import evaluate_separated_demands
from src.etl_bigquery import (
    DATA_DIR,
    fetch_cnes_rs_sus_monthly,
    fetch_inmet_rs_wind,
    fetch_rs_hydro_regions,
    fetch_sedec_rs_complete,
    fetch_sih_rs_sem_dengue,
    get_gcp_project_id,
)
from src.infra_events import load_infra_events_daily

warnings.filterwarnings("ignore")


def run_rs_urgencia_cids_pipeline_2024(
    climate_parquet: Path | str = DATA_DIR / "flood_dataset_br_2010_2025.parquet",
    cnes_area_parquet: Path | str = DATA_DIR / "cnes_ibge_area_br_2010_2025.parquet",
    epochs: int = 10,
    occupancy_target: float = 0.85,
    save_plots_dir: Optional[Path | str] = None,
    show_plots: bool = False,
    use_wind: bool = True,
    use_ana: bool = True,
    use_infra: bool = True,
    infra_csv: Path | str = DATA_DIR / "eventos_infraestrutura_rs.csv",
    model_save_path: Optional[Path | str] = None,
    model_load_path: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Executa a auditoria completa out-of-time de 2024 no RS."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_id = get_gcp_project_id()
    print(
        f"[DISPOSITIVO] {device.upper()} | Auditoria RS 2024 "
        f"(Vento={use_wind} | ANA={use_ana} | Infra={use_infra})"
    )

    df_reg = fetch_rs_hydro_regions(project_id)
    df_sedec = fetch_sedec_rs_complete(project_id)
    df_cnes_sus = fetch_cnes_rs_sus_monthly(project_id)
    df_sih = fetch_sih_rs_sem_dengue(project_id)

    df_wind = fetch_inmet_rs_wind(project_id) if use_wind else None
    df_ana = fetch_ana_rs_daily_cotas() if use_ana else None

    df_infra_daily = None
    if use_infra:
        df_clim_dates = pd.read_parquet(climate_parquet, columns=["data"])
        dates_idx = pd.to_datetime(df_clim_dates["data"]).dt.normalize()
        regioes = sorted(df_reg["regiao_hidro"].unique())
        df_infra_daily = load_infra_events_daily(dates_idx, regioes, csv_path=infra_csv)

    df_panel, dyn_cols = build_enriched_basin_panel(
        climate_parquet=climate_parquet,
        cnes_area_parquet=cnes_area_parquet,
        df_reg=df_reg,
        df_sedec=df_sedec,
        df_wind=df_wind,
        df_ana=df_ana,
        df_infra_daily=df_infra_daily,
    )
    df_pred_24 = train_and_predict_basin_2024(
        df_panel=df_panel,
        dyn_cols=dyn_cols,
        epochs=epochs,
        seq_len=21,
        device=device,
        model_save_path=model_save_path,
        model_load_path=model_load_path,
    )
    res_demandas = evaluate_separated_demands(
        df_panel=df_panel,
        df_pred_24=df_pred_24,
        df_reg=df_reg,
        df_sih=df_sih,
        df_cnes_sus=df_cnes_sus,
        occupancy_target=occupancy_target,
        save_plots_dir=save_plots_dir,
        show_plots=show_plots,
    )
    return df_pred_24, res_demandas


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auditoria RS 2024: Bacias + Vento Barra RG + ANA + Falhas Infra + SIH/SUS."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas para HydroFiLMRegionalBasinNet.")
    parser.add_argument(
        "--climate-parquet",
        type=Path,
        default=DATA_DIR / "flood_dataset_br_2010_2025.parquet",
        help="Caminho do Parquet climático diário.",
    )
    parser.add_argument(
        "--cnes-area-parquet",
        type=Path,
        default=DATA_DIR / "cnes_ibge_area_br_2010_2025.parquet",
        help="Caminho do Parquet CNES/IBGE/Área.",
    )
    parser.add_argument(
        "--occupancy-target",
        type=float,
        default=0.85,
        help="Taxa de ocupação segura alvo (padrão 0.85).",
    )
    parser.add_argument(
        "--save-plots-dir",
        type=Path,
        default=None,
        help="Diretório opcional para salvar gráficos PNG por coorte.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Exibe janelas interativas do Matplotlib ao finalizar cada coorte.",
    )
    parser.add_argument(
        "--no-wind",
        action="store_true",
        help="Desativa extração/uso de vento INMET (Barra de Rio Grande).",
    )
    parser.add_argument(
        "--no-ana",
        action="store_true",
        help="Desativa uso das cotas fluviométricas da ANA.",
    )
    parser.add_argument(
        "--no-infra",
        action="store_true",
        help="Desativa eventos binários de rompimento de diques e casas de bombas.",
    )
    parser.add_argument(
        "--infra-csv",
        type=Path,
        default=DATA_DIR / "eventos_infraestrutura_rs.csv",
        help="Caminho do CSV de eventos binários de falha de infraestrutura.",
    )
    parser.add_argument(
        "--ana-realtime-only",
        action="store_true",
        help="Consulta e exibe apenas o boletim atual em tempo real das estações ANA no RS.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Caminho .pt para salvar os pesos e estatísticas de normalização após o treino.",
    )
    parser.add_argument(
        "--load-model",
        type=Path,
        default=None,
        help="Caminho .pt de checkpoint salvo para rodar inferência sem retreinar a rede neural.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.ana_realtime_only:
        df_rt = fetch_realtime_rs_cotas()
        if df_rt.empty:
            print("[ANA TEMPO REAL] Sem leituras disponíveis.")
        else:
            print("\n=== BOLETIM TELEMETRIA ANA EM TEMPO REAL (RS) ===")
            print(df_rt.to_string(index=False))
    else:
        run_rs_urgencia_cids_pipeline_2024(
            climate_parquet=args.climate_parquet,
            cnes_area_parquet=args.cnes_area_parquet,
            epochs=args.epochs,
            occupancy_target=args.occupancy_target,
            save_plots_dir=args.save_plots_dir,
            show_plots=args.show_plots,
            use_wind=not args.no_wind,
            use_ana=not args.no_ana,
            use_infra=not args.no_infra,
            infra_csv=args.infra_csv,
            model_save_path=args.save_model,
            model_load_path=args.load_model,
        )
