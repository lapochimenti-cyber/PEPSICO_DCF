
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.fcff.model import build_params, fcff

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = BASE_DIR / "config" / "config.yaml"
DEFAULT_OUT = BASE_DIR / "output" / "FCFF_deterministico_v0.7_G2.csv"


def load_g_path(xlsx_path: Path, scenario: str) -> np.ndarray:
    """Legge i 10 tassi g_t dal foglio 'Scenari' di PEP_scenari.xlsx."""
    col_map = {
        "bear": "g_Basso",
        "base": "g_Base",
        "growth": "g_Alto",
    }
    df = pd.read_excel(xlsx_path, sheet_name="Scenari")
    return df[col_map[scenario]].dropna().to_numpy(dtype=float)[:10]


def run_deterministic_case(cfg: dict, scenario: str, s_inf_override: float = None):
    """Esegue il calcolo deterministico (sigma=0) per uno scenario leggendo dal config."""
    fcff_cfg = dict(cfg["fcff_parameters"])
    fcff_cfg["m_growth"] = cfg["fcff_parameters"]["m_growth"][scenario]
    fcff_cfg["m_inf"] = float(cfg["fcff_parameters"]["m_inf"][scenario])

    if s_inf_override is not None:
        fcff_cfg["s_inf"] = float(s_inf_override)
    else:
        fcff_cfg["s_inf"] = float(cfg["fcff_parameters"]["s_inf"][scenario])

    p = build_params(fcff_cfg)

    r0 = float(cfg["revenues"]["r0"])
    d05_path = BASE_DIR / cfg["revenues"]["d05"]["source"]
    g = load_g_path(d05_path, scenario)

    R = r0 * np.cumprod(1.0 + g)
    nopat, reinv, flow = fcff(R[None, :], r0, p)

    return fcff_cfg, p, g, R, nopat[0], reinv[0], flow[0]


def export_deterministic_csv(
    config_path: Path = DEFAULT_CONFIG,
    out_path: Path = DEFAULT_OUT,
) -> Path:
    """Genera il CSV deterministico sui 3 scenari + i 3 casi di sensitività spread."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    scenarios = list(cfg["fcff_parameters"]["s_inf"].keys())  # ['bear', 'base', 'growth']
    s_sens = float(cfg["fcff_parameters"]["s_sensitivity"])

    cases = [(sc, None, sc) for sc in scenarios] + [
        (sc, s_sens, f"{sc}_spread5") for sc in scenarios
    ]

    rows = []
    for scenario, s_override, label in cases:
        a, p, g, R, nopat, reinv, flow = run_deterministic_case(cfg, scenario, s_override)
        roic = float(a["w_inf"]) + float(a["s_inf"])
        for t in range(10):
            rows.append([
                label,
                t + 1,
                2026 + t,
                f"{g[t]:.6f}",
                f"{p['margine'][t]:.6f}",
                f"{p['aliquota'][t]:.6f}",
                f"{R[t]:.4f}",
                f"{nopat[t]:.4f}",
                f"{reinv[t]:.4f}",
                f"{flow[t]:.4f}",
                f"{a['m_inf']:.4f}",
                f"{roic:.6f}",
                f"{p['rr_inf']:.6f}",
            ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario", "t", "anno_fiscale", "crescita_quota",
            "margine_EBIT_quota", "aliquota_quota", "ricavi_USD_mld",
            "NOPAT_USD_mld", "reinvestimento_USD_mld", "FCFF_USD_mld",
            "m_inf_quota", "ROIC_inf_quota", "rr_inf_quota",
        ])
        w.writerows(rows)

    print(f"written {out_path} ({len(rows)} rows)")
    return out_path


if __name__ == "__main__":
    export_deterministic_csv()