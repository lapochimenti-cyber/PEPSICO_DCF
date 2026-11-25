import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from src.revenues.model import simulate_revenues, estimate_sigma, load_g_path_from_d05
from src.revenues.calibration import calibrate_sigma
from src.fcff.model import fcff, build_params
from src.rates.model import compute_enterprise_value
from src.bridge.model import compute_equity_and_price

def load_g_path(xlsx_path: Path, scenario: str) -> np.ndarray:
    """Legge i 10 tassi g_t dal foglio 'Scenari' di PEP_scenari.xlsx."""
    col_map = {
        "bear": "g_Basso",
        "base": "g_Base",
        "growth": "g_Alto",
    }
    if scenario not in col_map:
        raise ValueError(f"Scenario non valido: {scenario}. Usa {list(col_map.keys())}")
    
    df = pd.read_excel(xlsx_path, sheet_name="Scenari")
    return df[col_map[scenario]].dropna().to_numpy(dtype=float)[:10]

def run_pipeline(config_path: str, scenario: str = "base", is_final_run: bool = False, override_sigma: float = None, override_s_inf: float = None) -> dict:
  
  base_dir = Path(__file__).resolve().parent.parent
  with open(base_dir / config_path, "r") as f:
    cfg = yaml.safe_load(f)
  
  # the final run needs to stop if the data are synthetic or temporary
  status = cfg["meta"]["status"]

  if is_final_run and status != "real":
    raise RuntimeError("Pipeline block: all data need to be in status 'real'!")

  seed = int(cfg["simulation"]["seed"])
  n_paths = int(cfg["simulation"]["n_paths"])
  r0 = float(cfg["revenues"]["r0"])
  d05_path = base_dir / cfg["revenues"]["d05"]["source"]
  g_path = load_g_path(d05_path, scenario)

  fcff_cfg = dict(cfg["fcff_parameters"])
  fcff_cfg["s_inf"] = float(cfg["fcff_parameters"]["s_inf"][scenario]) if override_s_inf is None else float(override_s_inf)
  fcff_cfg["m_growth"] = list(cfg["fcff_parameters"]["m_growth"][scenario])
  fcff_cfg["m_inf"] = float(cfg["fcff_parameters"]["m_inf"][scenario])

  rates_cfg = dict(cfg["rates_and_terminal"])
  if not np.isclose(fcff_cfg["w_inf"], rates_cfg["wacc_maturity"], rtol=0, atol=1e-12):
    raise ValueError("G2 e G3 devono usare lo stesso WACC di maturità")
  # Stessa ipotesi di rendimento nel reinvestimento G2 e nel terminal value.
  rates_cfg["roic_terminal"] = float(fcff_cfg["w_inf"] + fcff_cfg["s_inf"])


  calibration = calibrate_sigma(cfg["revenues"])
  sigma = calibration["sigma"] if override_sigma is None else float(override_sigma)
  if not np.isfinite(sigma) or sigma < 0:
    raise ValueError("Sigma must be finite and non-negative")

  # chain execution G1->G2->G3, the name are provvisory: we will implement the right functions once we have them
  revs = simulate_revenues(r0 = r0, g_path = g_path, sigma = sigma, n_paths = n_paths, seed = seed)
  fcff_out = fcff(ricavi = revs, ricavo_base = r0, params = build_params(fcff_cfg))
  ev_out = compute_enterprise_value(revenues = revs, fcff = fcff_out[2], params = rates_cfg, nopat_h1 = fcff_out[0][:, -1] * (1 + fcff_cfg["g_inf"]), w_inf = fcff_cfg["w_inf"], s_inf = fcff_cfg["s_inf"])
  bridge_out = compute_equity_and_price(ev = ev_out["ev"], bridge_params = cfg["bridge"], is_final_run = is_final_run)

  prices = bridge_out["share_price"]
  evs = ev_out["ev"]

  summary = {
    "status": status,
    "seed": seed,
    "n_paths": n_paths,
    "sigma_used": sigma,
    "price_mean": float(np.mean(prices)),
    "price_median": float(np.median(prices)),
    "price_p5": float(np.percentile(prices, 5.0)),
    "price_p95": float(np.percentile(prices, 95.0)),
    "price_std_err": float(np.std(prices) / np.sqrt(n_paths)),
    "ev_mean": float(np.mean(evs)),
    "ev_median": float(np.median(evs)),
    "ev_std_err": float(np.std(evs) / np.sqrt(n_paths)),
    "mean_tv_weight": float(np.mean(ev_out["tv_weight"])),
    "negative_equity_count": int(bridge_out["negative_equity_count"]),
    "revenues_matrix": revs,
    "prices_array": prices

  }


  return summary


if __name__ == "__main__":
    scenarios = ["bear", "base", "growth"]
    results_by_scenario = {}

    for sc in scenarios:
        res = run_pipeline(config_path = "config/config.yaml", scenario = sc, is_final_run = False)
        results_by_scenario[sc] = res

        print("=" * 50)
        print(f"SCENARIO: {sc.upper()} | Stato: {res['status']} | Seed: {res['seed']}")
        print("=" * 50)
        print(f"Prezzo Medio:     ${res['price_mean']:.2f}")
        print(f"Prezzo Mediano:   ${res['price_median']:.2f}")
        print(f"Intervallo P5-95: [${res['price_p5']:.2f} - ${res['price_p95']:.2f}]")
        print(f"EV Medio:         ${res['ev_mean']:.2f}B (SE MC: ${res['ev_std_err']:.4f}B)")
        print(f"Peso medio TV:    {res['mean_tv_weight']*100:.1f}%")


    config_path = "config/config.yaml"
    base_dir = Path(__file__).resolve().parent.parent
    with open(base_dir / config_path, "r") as f:
        cfg = yaml.safe_load(f)

    s_sens = float(cfg["fcff_parameters"]["s_sensitivity"])
    res_sens = run_pipeline(
        config_path = "config/config.yaml",
        scenario = "base",
        is_final_run = False,
        override_s_inf = s_sens
    )
    print("=" * 60)
    print(f"SENSITIVITY (BASE + s_inf={s_sens:.2%}) | Seed: {res_sens['seed']}")
    print("=" * 60)
    print(f"Prezzo Medio:     ${res_sens['price_mean']:.2f}")
    print(f"Prezzo Mediano:   ${res_sens['price_median']:.2f}")
    print(f"Intervallo P5-95: [${res_sens['price_p5']:.2f} - ${res_sens['price_p95']:.2f}]")
