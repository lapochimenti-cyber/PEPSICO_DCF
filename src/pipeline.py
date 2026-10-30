from src.revenues.model import simulate_revenues, estimate_sigma
from src.fcff.model import fcff
from src.rates.model import ???
from src.bridge.model import ???


def run_pipeline(cfg: dict, is_final_run: bool = False, override_sigma: float = None, g3_functions: dict = None) -> dict:

  # the final run needs to stop if the data are synthetic or temporary
  status = cfg["meta"]["status"]

  if is_final_run and status != "real":
    raise RuntimeError("Pipeline block: all data need to be in status 'real'!")

  if g3_functions is None:
    raise RuntimeError("Pending G3 repo functions: connect G3_FUNCTIONS before running")

  for name in ("build_wacc_path", "compute_terminal_value", "compute_equity_and_price"):
    if not callable(g3_functions.get(name)):
      raise ValueError(f"Missing G3 adapter: {name}")

  if any(cfg["rates_and_terminal"][key].get(stage) is None
         for key in ("stage_years", "wacc_by_stage")
         for stage in ("growth", "transition", "maturity")):
    raise ValueError("Pending G2/G3: stage durations and WACC values are required")

  seed = int(cfg["simulation"]["seed"])
  n_paths = int(cfg["simulation"]["n_paths"])
  r0 = float(cfg["revenues"]["r0"])
  g_path = np.asarray(cfg["revenues"]["g_path"], dtype = float)
  sigma = float(cfg["revenues"]["sigma"]) if override_sigma is None else override_sigma

  # chain execution G1->G2->G3, the name are provvisory: we will implement the right functions once we have them
  revs = simulate_revenues(r0 = r0, g_path = g_path, sigma = sigma, n_paths = n_paths, seed = seed)
  fcff_out = fcff(ricavi = revs, r0, params = cfg["fcff_parameters"])
  ev_out = compute_enterprise_value(revenues = revs, fcff = fcff_out["fcff"], rates_and_terminal = cfg["rates_and_terminal"], g3_functions = g3_functions)
  bridge_out = compute_equity_and_price(ev = ev_out["enterprise_value"], bridge_parameters = cfg["bridge"], g3_functions = g3_functions)

  prices = bridge_out["share_price"]
  evs = ev_out["enterprise_value"]

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

