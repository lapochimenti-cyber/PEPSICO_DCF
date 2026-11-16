import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from src.revenues.model import simulate_revenues, estimate_sigma, load_g_path_from_d05
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

def run_pipeline(config_path: str, scenario: str = "base", is_final_run: bool = False, override_sigma: float = None) -> dict:
  
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
  sigma = float(cfg["revenues"]["sigma"]) if override_sigma is None else override_sigma
  fcff_cfg = dict(cfg["fcff_parameters"])
  fcff_cfg["s_inf"] = float(cfg["fcff_parameters"]["s_inf"][scenario])

  # chain execution G1->G2->G3, the name are provvisory: we will implement the right functions once we have them
  revs = simulate_revenues(r0 = r0, g_path = g_path, sigma = sigma, n_paths = n_paths, seed = seed)
  fcff_out = fcff(ricavi = revs, ricavo_base = r0, params = build_params(fcff_cfg))
  ev_out = compute_enterprise_value(revenues = revs, fcff = fcff_out[2], params = cfg["rates_and_terminal"], nopat_h1 = fcff_out[0][:, -1] * (1 + cfg["fcff_parameters"]["g_inf"]))
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

