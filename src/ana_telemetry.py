"""Integração com a API HidroWebService (ANA) para cotas fluviométricas históricas e em tempo real."""

from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from src.etl_bigquery import DATA_DIR, get_secret

DEFAULT_ANA_BASE_URL = "https://www.ana.gov.br/hidrowebservice"

# Estações fluviométricas de referência da Rede Hidrometeorológica Nacional (ANA/SGB) no RS
ANA_RS_STATIONS: Dict[str, List[Tuple[str, str]]] = {
    "Metropolitana e Bacia do Guaíba": [
        ("87450004", "Cais Mauá C6 - Guaíba"),
        ("87444000", "Usina do Gasômetro - Guaíba"),
        ("87382000", "São Leopoldo - Rio dos Sinos"),
    ],
    "Vales (Taquari e Rio Pardo)": [
        ("86879300", "Estrela - Rio Taquari"),
        ("86510000", "Muçum - Rio Taquari"),
        ("85900000", "Rio Pardo - Rio Jacuí"),
    ],
    "Serra Gaúcha (Alto Taquari/Caí)": [
        ("87170000", "Barca do Caí - Rio Caí"),
    ],
    "Depressão Central (Médio Jacuí)": [
        ("85400000", "Dona Francisca - Rio Jacuí"),
    ],
    "Região Sul e Lagoa dos Patos": [
        ("87905000", "Passo do Mendonça - Rio Camaquã"),
    ],
    "Fronteira Oeste e Campanha": [
        ("77150000", "Uruguaiana - Rio Uruguai"),
    ],
    "Noroeste e Missões": [
        ("75230000", "Santo Ângelo - Rio Ijuí"),
        ("75155000", "Passo Faxinal - Rio Ijuí"),
    ],
    "Planalto Norte (Alto Jacuí/Uruguai)": [
        ("85400000", "Dona Francisca - Alto/Médio Jacuí"),
    ],
}

EMPTY_ANA_DAILY_COLS: List[str] = [
    "regiao_hidro",
    "data",
    "cota_max_cm",
    "cota_media_cm",
    "cota_var_24h_cm",
    "cota_var_72h_cm",
]


def _extract_records(payload: Any) -> List[Dict[str, Any]]:
    """Extrai lista de registros de payloads JSON variados da HidroWebService."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("items", "Items", "data", "Data", "registros", "Registros", "result"):
            inner = payload.get(key)
            if isinstance(inner, list):
                return [r for r in inner if isinstance(r, dict)]
            if isinstance(inner, dict):
                nested = _extract_records(inner)
                if nested:
                    return nested
    return []


def _extract_token(payload: Any) -> Optional[str]:
    """Localiza o token JWT retornado pelo endpoint OAuth da ANA."""
    if not isinstance(payload, dict):
        return None
    for key in ("tokenautenticacao", "TokenAutenticacao", "accessToken", "access_token", "token", "Token"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    for nested_key in ("items", "data", "result"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            tok = _extract_token(nested)
            if tok:
                return tok
    return None


def _parse_float_br(value: Any) -> float:
    """Converte strings numéricas com vírgula ou ponto para float."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in ("nan", "null", "none", "-"):
        return np.nan
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


class ANAHidroClient:
    """Cliente REST autenticado para a API HidroWebService da ANA com cache de token."""

    def __init__(
        self,
        identificador: Optional[str] = None,
        senha: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
    ):
        self.identificador = identificador or get_secret("ANA_IDENTIFICADOR")
        self.senha = senha or get_secret("ANA_SENHA")
        self.base_url = (base_url or get_secret("ANA_API_BASE_URL", DEFAULT_ANA_BASE_URL) or DEFAULT_ANA_BASE_URL).rstrip("/")
        self.timeout = timeout
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._session = requests.Session()

    @property
    def has_credentials(self) -> bool:
        return bool(self.identificador and self.senha)

    def _authenticate(self, force: bool = False) -> str:
        """Obtém e faz cache do token OAuth da ANA por 50 minutos."""
        if not self.has_credentials:
            raise ValueError(
                "Credenciais da ANA ausentes. Defina 'ANA_IDENTIFICADOR' e 'ANA_SENHA' no .env "
                "ou nos Secrets do Colab."
            )
        now = time.time()
        if not force and self._token and now < self._token_expires_at:
            return self._token

        url_auth = f"{self.base_url}/EstacoesTelemetricas/OAUth/v1"
        headers = {
            "identificador": self.identificador,
            "senha": self.senha,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        last_err: Optional[Exception] = None
        attempts = [
            ("GET", {"headers": headers}),
            ("POST", {"headers": headers, "json": {"identificador": self.identificador, "senha": self.senha}}),
            ("GET", {"headers": {"Accept": "application/json"}, "params": {"Identificador": self.identificador, "Senha": self.senha}}),
        ]
        for method, kwargs in attempts:
            try:
                resp = self._session.request(method, url_auth, timeout=self.timeout, **kwargs)
                if resp.status_code == 200:
                    token = _extract_token(resp.json())
                    if token:
                        self._token = token
                        self._token_expires_at = now + 3000.0  # 50 minutos (validade padrão de 60 min)
                        return token
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:250]}")
            except Exception as exc:
                last_err = exc

        raise RuntimeError(f"Falha ao autenticar na API HidroWebService da ANA: {last_err}")

    def _request_json(self, endpoint_path: str, params: Dict[str, Any]) -> Any:
        token = self._authenticate()
        url = f"{self.base_url}/{endpoint_path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        resp = self._session.get(url, headers=headers, params=params, timeout=self.timeout)
        if resp.status_code == 401:
            token = self._authenticate(force=True)
            headers["Authorization"] = f"Bearer {token}"
            resp = self._session.get(url, headers=headers, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def fetch_station_telemetry(
        self,
        codigo_estacao: str,
        data_inicio: str,
        data_fim: str,
    ) -> pd.DataFrame:
        """Consulta telemetria em alta frequência (cota em cm) para uma estação ANA."""
        empty_df = pd.DataFrame(columns=["codigo_estacao", "data_hora", "cota_cm"])
        params_variants = [
            {
                "Código da Estação": codigo_estacao,
                "Tipo Filtro Data": "DATA_LEITURA",
                "Data Busca (yyyy-MM-dd)": data_inicio,
                "Range Intervalo de busca": "DIAS_30",
            },
            {
                "codigoEstacao": codigo_estacao,
                "dataInicio": data_inicio,
                "dataFim": data_fim,
            },
            {
                "codEstacao": codigo_estacao,
                "dataInicio": data_inicio,
                "dataFim": data_fim,
            },
        ]

        records: List[Dict[str, Any]] = []
        for params in params_variants:
            try:
                payload = self._request_json(
                    "EstacoesTelemetricas/HidroinfoanaSerieTelemetricaDetalhada/v1",
                    params=params,
                )
                records = _extract_records(payload)
                if records:
                    break
            except Exception:
                continue

        if not records:
            return empty_df

        df_raw = pd.DataFrame(records)
        col_map_lower = {c.lower(): c for c in df_raw.columns}
        ts_col = next(
            (
                col_map_lower[k]
                for k in ("data_hora_medicao", "datahora", "data_hora", "datamedicao", "data")
                if k in col_map_lower
            ),
            None,
        )
        cota_col = next(
            (
                col_map_lower[k]
                for k in ("cota_adotada", "cota_sensor", "cota", "nivel", "cota_telemetrica")
                if k in col_map_lower
            ),
            None,
        )
        if not ts_col or not cota_col:
            return empty_df

        out = pd.DataFrame(
            {
                "codigo_estacao": str(codigo_estacao).strip(),
                "data_hora": pd.to_datetime(df_raw[ts_col], errors="coerce"),
                "cota_cm": df_raw[cota_col].map(_parse_float_br),
            }
        )
        out = out.dropna(subset=["data_hora", "cota_cm"])
        start_dt = pd.to_datetime(data_inicio)
        end_dt = pd.to_datetime(data_fim) + pd.Timedelta(days=1)
        out = out[(out["data_hora"] >= start_dt) & (out["data_hora"] < end_dt)]
        return out.sort_values("data_hora").reset_index(drop=True)

    def fetch_station_historical_cotas(
        self,
        codigo_estacao: str,
        data_inicio: str,
        data_fim: str,
    ) -> pd.DataFrame:
        """Consulta série diária histórica de cotas (via HidroSerieCotas ou janelas de telemetria)."""
        empty_df = pd.DataFrame(columns=["codigo_estacao", "data", "cota_media_cm", "cota_max_cm"])
        for endpoint in ("HidroSerieCotas/v1", "EstacoesTelemetricas/HidroSerieCotas/v1"):
            try:
                payload = self._request_json(
                    endpoint,
                    params={
                        "codEstacao": codigo_estacao,
                        "codigoEstacao": codigo_estacao,
                        "dataInicio": data_inicio,
                        "dataFim": data_fim,
                    },
                )
                records = _extract_records(payload)
                if records:
                    df_parsed = _parse_hidro_cotas_records(records, codigo_estacao)
                    if not df_parsed.empty:
                        return df_parsed
            except Exception:
                continue

        # Fallback: varredura em blocos de 30 dias na telemetria detalhada
        chunks: List[pd.DataFrame] = []
        cur = pd.to_datetime(data_inicio)
        end = pd.to_datetime(data_fim)
        while cur <= end:
            chunk_end = min(end, cur + pd.Timedelta(days=29))
            df_tel = self.fetch_station_telemetry(
                codigo_estacao=codigo_estacao,
                data_inicio=cur.strftime("%Y-%m-%d"),
                data_fim=chunk_end.strftime("%Y-%m-%d"),
            )
            if not df_tel.empty:
                df_tel["data"] = df_tel["data_hora"].dt.normalize()
                daily = df_tel.groupby("data", as_index=False).agg(
                    cota_media_cm=("cota_cm", "mean"),
                    cota_max_cm=("cota_cm", "max"),
                )
                daily["codigo_estacao"] = str(codigo_estacao).strip()
                chunks.append(daily[["codigo_estacao", "data", "cota_media_cm", "cota_max_cm"]])
            cur = chunk_end + pd.Timedelta(days=1)

        if not chunks:
            return empty_df
        return pd.concat(chunks, ignore_index=True).drop_duplicates(["codigo_estacao", "data"])


def _parse_hidro_cotas_records(records: List[Dict[str, Any]], codigo_estacao: str) -> pd.DataFrame:
    """Converte registros mensais (Cota01..Cota31) ou diários da HidroWebService em série diária."""
    df_raw = pd.DataFrame(records)
    if df_raw.empty:
        return pd.DataFrame(columns=["codigo_estacao", "data", "cota_media_cm", "cota_max_cm"])

    cota_day_cols = [f"Cota{d:02d}" for d in range(1, 32) if f"Cota{d:02d}" in df_raw.columns]
    if cota_day_cols and ("Data" in df_raw.columns or "Data_Hora_Medicao" in df_raw.columns):
        dt_col = "Data" if "Data" in df_raw.columns else "Data_Hora_Medicao"
        rows = []
        for _, row in df_raw.iterrows():
            base_dt = pd.to_datetime(row[dt_col], dayfirst=True, errors="coerce")
            if pd.isna(base_dt):
                continue
            for day_idx in range(1, 32):
                col = f"Cota{day_idx:02d}"
                if col not in df_raw.columns:
                    continue
                val = _parse_float_br(row[col])
                if np.isnan(val):
                    continue
                try:
                    dt = base_dt.replace(day=day_idx).normalize()
                    rows.append((str(codigo_estacao).strip(), dt, val))
                except ValueError:
                    continue
        if not rows:
            return pd.DataFrame(columns=["codigo_estacao", "data", "cota_media_cm", "cota_max_cm"])
        df_daily = pd.DataFrame(rows, columns=["codigo_estacao", "data", "cota_cm"])
        return df_daily.groupby(["codigo_estacao", "data"], as_index=False).agg(
            cota_media_cm=("cota_cm", "mean"),
            cota_max_cm=("cota_cm", "max"),
        )

    col_map = {c.lower(): c for c in df_raw.columns}
    dt_col = next((col_map[k] for k in ("data", "data_medicao", "datahora") if k in col_map), None)
    cota_col = next((col_map[k] for k in ("cota", "cota_cm", "cota_media", "nivel") if k in col_map), None)
    if not dt_col or not cota_col:
        return pd.DataFrame(columns=["codigo_estacao", "data", "cota_media_cm", "cota_max_cm"])

    df_out = pd.DataFrame(
        {
            "codigo_estacao": str(codigo_estacao).strip(),
            "data": pd.to_datetime(df_raw[dt_col], dayfirst=True, errors="coerce").dt.normalize(),
            "cota_cm": df_raw[cota_col].map(_parse_float_br),
        }
    ).dropna(subset=["data", "cota_cm"])
    return df_out.groupby(["codigo_estacao", "data"], as_index=False).agg(
        cota_media_cm=("cota_cm", "mean"),
        cota_max_cm=("cota_cm", "max"),
    )


def _aggregate_daily_to_regions(
    df_station_daily: pd.DataFrame,
    stations: Optional[Dict[str, List[Tuple[str, str]]]] = None,
) -> pd.DataFrame:
    """Agrega cotas diárias por região hidrográfica com variações em 24h e 72h."""
    if df_station_daily.empty:
        return pd.DataFrame(columns=EMPTY_ANA_DAILY_COLS)

    station_map = stations or ANA_RS_STATIONS
    rows_map = [
        {"codigo_estacao": str(cod).strip(), "regiao_hidro": reg}
        for reg, st_list in station_map.items()
        for cod, _ in st_list
    ]
    df_map = pd.DataFrame(rows_map).drop_duplicates()

    merged = df_station_daily.copy()
    merged["codigo_estacao"] = merged["codigo_estacao"].astype(str).str.strip()
    merged["data"] = pd.to_datetime(merged["data"]).dt.normalize()
    merged = merged.merge(df_map, on="codigo_estacao", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=EMPTY_ANA_DAILY_COLS)

    reg_daily = (
        merged.groupby(["regiao_hidro", "data"], as_index=False)
        .agg(
            cota_max_cm=("cota_max_cm", "max"),
            cota_media_cm=("cota_media_cm", "mean"),
        )
        .sort_values(["regiao_hidro", "data"])
        .reset_index(drop=True)
    )
    grp = reg_daily.groupby("regiao_hidro")["cota_max_cm"]
    reg_daily["cota_var_24h_cm"] = grp.diff(1).fillna(0.0)
    reg_daily["cota_var_72h_cm"] = grp.diff(3).fillna(0.0)
    return reg_daily[EMPTY_ANA_DAILY_COLS]


def fetch_realtime_rs_cotas(
    hours_back: int = 72,
    stations: Optional[Dict[str, List[Tuple[str, str]]]] = None,
) -> pd.DataFrame:
    """Consulta cotas fluviométricas em tempo real das estações de referência no RS (API ANA)."""
    cols = [
        "regiao_hidro",
        "codigo_estacao",
        "nome_estacao",
        "cota_atual_cm",
        "cota_var_24h_cm",
        "cota_max_janela_cm",
        "ultima_leitura",
    ]
    client = ANAHidroClient()
    if not client.has_credentials:
        print(
            "[ANA TEMPO REAL] Credenciais ANA_IDENTIFICADOR/ANA_SENHA não configuradas. "
            "Telemetria em tempo real ignorada."
        )
        return pd.DataFrame(columns=cols)

    station_map = stations or ANA_RS_STATIONS
    now = datetime.now()
    start_dt = now - timedelta(hours=hours_back)
    data_inicio = start_dt.strftime("%Y-%m-%d")
    data_fim = now.strftime("%Y-%m-%d")

    seen_codes: Dict[str, pd.DataFrame] = {}
    rows = []
    for reg, st_list in station_map.items():
        for cod, nome in st_list:
            if cod not in seen_codes:
                try:
                    seen_codes[cod] = client.fetch_station_telemetry(cod, data_inicio, data_fim)
                except Exception as exc:
                    print(f"[ANA TEMPO REAL] Aviso ao consultar estação {cod} ({nome}): {exc}")
                    seen_codes[cod] = pd.DataFrame(columns=["codigo_estacao", "data_hora", "cota_cm"])
            df_tel = seen_codes[cod]
            if df_tel.empty:
                continue
            df_tel = df_tel.sort_values("data_hora")
            latest = df_tel.iloc[-1]
            cutoff_24h = latest["data_hora"] - pd.Timedelta(hours=24)
            older = df_tel[df_tel["data_hora"] <= cutoff_24h]
            ref_24h = older.iloc[-1]["cota_cm"] if not older.empty else df_tel.iloc[0]["cota_cm"]
            rows.append(
                {
                    "regiao_hidro": reg,
                    "codigo_estacao": cod,
                    "nome_estacao": nome,
                    "cota_atual_cm": float(latest["cota_cm"]),
                    "cota_var_24h_cm": float(latest["cota_cm"] - ref_24h),
                    "cota_max_janela_cm": float(df_tel["cota_cm"].max()),
                    "ultima_leitura": latest["data_hora"],
                }
            )
    return pd.DataFrame(rows, columns=cols)


def fetch_ana_rs_daily_cotas(
    cache_path: Path | str = DATA_DIR / "ana_rs_cotas_diarias.parquet",
    ano_inicio: int = 2010,
    ano_fim: int = 2025,
    include_realtime_tail: bool = True,
) -> pd.DataFrame:
    """Carrega série diária de cotas ANA por região (cache, API HidroWebService ou fallback vazio)."""
    path = Path(cache_path)
    client = ANAHidroClient()

    if path.exists():
        df_cached = pd.read_parquet(path)
        if len(df_cached) > 0:
            df_cached["data"] = pd.to_datetime(df_cached["data"]).dt.normalize()
            if include_realtime_tail and client.has_credentials:
                try:
                    df_rt = fetch_realtime_rs_cotas(hours_back=120)
                    if not df_rt.empty:
                        df_rt["data"] = pd.to_datetime(df_rt["ultima_leitura"]).dt.normalize()
                        rt_reg = df_rt.groupby(["regiao_hidro", "data"], as_index=False).agg(
                            cota_max_cm=("cota_max_janela_cm", "max"),
                            cota_media_cm=("cota_atual_cm", "mean"),
                            cota_var_24h_cm=("cota_var_24h_cm", "mean"),
                        )
                        rt_reg["cota_var_72h_cm"] = np.nan
                        merged = (
                            pd.concat([df_cached, rt_reg], ignore_index=True)
                            .drop_duplicates(["regiao_hidro", "data"], keep="last")
                            .sort_values(["regiao_hidro", "data"])
                            .reset_index(drop=True)
                        )
                        grp = merged.groupby("regiao_hidro")["cota_max_cm"]
                        merged["cota_var_24h_cm"] = merged["cota_var_24h_cm"].fillna(grp.diff(1)).fillna(0.0)
                        merged["cota_var_72h_cm"] = merged["cota_var_72h_cm"].fillna(grp.diff(3)).fillna(0.0)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        merged.to_parquet(path, index=False)
                        return merged[EMPTY_ANA_DAILY_COLS]
                except Exception as exc:
                    print(f"[ANA] Aviso ao complementar cauda em tempo real: {exc}")
            return df_cached[EMPTY_ANA_DAILY_COLS]
        path.unlink()

    if not client.has_credentials:
        print(
            "[ANA] Cache de cotas não encontrado e credenciais ANA_IDENTIFICADOR/ANA_SENHA ausentes. "
            "Configure o .env (ou Secrets do Colab) ou importe CSVs do portal HidroWeb via "
            "import_hidroweb_csv(). Prosseguindo sem cotas ANA."
        )
        return pd.DataFrame(columns=EMPTY_ANA_DAILY_COLS)

    data_inicio = f"{ano_inicio}-01-01"
    data_fim = min(datetime.now().strftime("%Y-%m-%d"), f"{ano_fim}-12-31")
    station_dfs: List[pd.DataFrame] = []
    unique_codes = sorted({cod for st_list in ANA_RS_STATIONS.values() for cod, _ in st_list})
    for cod in unique_codes:
        try:
            df_st = client.fetch_station_historical_cotas(cod, data_inicio=data_inicio, data_fim=data_fim)
            if not df_st.empty:
                station_dfs.append(df_st)
        except Exception as exc:
            print(f"[ANA] Falha ao baixar histórico da estação {cod}: {exc}")

    if not station_dfs:
        print("[ANA] Nenhuma série fluviométrica retornada pela API. Prosseguindo sem cotas ANA.")
        return pd.DataFrame(columns=EMPTY_ANA_DAILY_COLS)

    reg_daily = _aggregate_daily_to_regions(pd.concat(station_dfs, ignore_index=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    reg_daily.to_parquet(path, index=False)
    print(f"[ANA] Salvas {len(reg_daily):,} linhas região-dia em {path}.")
    return reg_daily


def import_hidroweb_csv(
    csv_dir: Path | str,
    cache_path: Path | str = DATA_DIR / "ana_rs_cotas_diarias.parquet",
) -> pd.DataFrame:
    """Importa arquivos CSV baixados manualmente do HidroWeb (cotas_*.csv) para cache Parquet."""
    base_dir = Path(csv_dir)
    csv_files = sorted(base_dir.glob("cotas_*.csv")) or sorted(base_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum arquivo CSV do HidroWeb encontrado em {base_dir}.")

    station_dfs: List[pd.DataFrame] = []
    for csv_file in csv_files:
        df_raw = None
        for enc in ("utf-8", "latin-1"):
            try:
                df_raw = pd.read_csv(
                    csv_file,
                    sep=";",
                    decimal=",",
                    comment="/",
                    encoding=enc,
                    on_bad_lines="skip",
                )
                if "EstacaoCodigo" in df_raw.columns or "Data" in df_raw.columns:
                    break
            except Exception:
                df_raw = None
        if df_raw is None or df_raw.empty:
            continue

        cod_col = next((c for c in df_raw.columns if "estacaocodigo" in c.lower()), None)
        if cod_col:
            for cod, sub in df_raw.groupby(cod_col):
                parsed = _parse_hidro_cotas_records(sub.to_dict("records"), str(cod).strip())
                if not parsed.empty:
                    station_dfs.append(parsed)
        else:
            stem_digits = "".join(ch for ch in csv_file.stem if ch.isdigit())
            if stem_digits:
                parsed = _parse_hidro_cotas_records(df_raw.to_dict("records"), stem_digits)
                if not parsed.empty:
                    station_dfs.append(parsed)

    if not station_dfs:
        raise ValueError(f"Não foi possível extrair cotas válidas dos CSVs em {base_dir}.")

    reg_daily = _aggregate_daily_to_regions(pd.concat(station_dfs, ignore_index=True))
    out_path = Path(cache_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reg_daily.to_parquet(out_path, index=False)
    print(f"[ANA CSV] Importadas {len(reg_daily):,} linhas região-dia para {out_path}.")
    return reg_daily
