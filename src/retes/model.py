# Hey there G3, it's your turn to write some code now! Good work...
import numpy as np

# Local adapter contract only: actual G3 module names/signatures are pending.
# G3 functions must be connected explicitly; no interpolation or TV formula
# is assumed here.

def compute_enterprise_value(revenues: np.ndarray, fcff: np.ndarray, rates_and_terminal: dict, g3_functions: dict) -> dict:

  stages = ("growth", "transition", "maturity")
  stage_years = rates_and_terminal["stage_years"]
  wacc_by_stage = rates_and_terminal["wacc_by_stage"]

  if any(stage_years.get(s) is None or wacc_by_stage.get(s) is None for s in stages):
    raise ValueError("Pending G2/G3: provide the duration and WACC of each stage")

  if any(isinstance(stage_years[s], bool) or
         not isinstance(stage_years[s], (int, np.integer)) or
         stage_years[s] < 0 for s in stages):
    raise ValueError("Stage durations must be non-negative integer years")

  if sum(stage_years[s] for s in stages) != fcff.shape[1]:
    raise ValueError("Stage durations must sum to the forecast horizon")
  stage_rates = np.asarray([wacc_by_stage[s] for s in stages], dtype=float)

  if not np.all(np.isfinite(stage_rates)) or np.any(stage_rates <= -1.0):
    raise ValueError("Stage WACCs must be finite decimal rates greater than -1")

  # Adapter to G3's vector builder; one annual vector for every simulation path.
  wacc_path = np.asarray(g3_functions["build_wacc_path"](
      wacc_by_stage=wacc_by_stage, stage_years=stage_years), dtype=float)

  if wacc_path.shape != (fcff.shape[1],):
    raise ValueError("G3 must return one WACC per year, shared by all paths")

  if not np.all(np.isfinite(wacc_path)) or np.any(wacc_path <= -1.0):
    raise ValueError("Annual WACCs must be finite decimal rates greater than -1")

  # Annual nominal USD rates; year-end discounting starts at t=1.
  df_path = 1.0 / np.cumprod(1.0 + wacc_path)
  ev0 = np.sum(fcff * df_path[None, :], axis=1)

  # Local adapter must return UNDISCOUNTED TV at H, one value per path.
  # Adapt G3's actual output to this convention when their function arrives.
  # Maturity WACC is the sole source for the terminal discount rate.

  tv_h = np.asarray(g3_functions["compute_terminal_value"](
      revenues=revenues, fcff=fcff,
      wacc_terminal=float(wacc_by_stage["maturity"]),
      terminal_parameters=rates_and_terminal), dtype=float)

  if tv_h.shape != (fcff.shape[0],) or not np.all(np.isfinite(tv_h)):
    raise ValueError("G3 terminal adapter must return one finite TV at H per path")

  ev_tv = tv_h * df_path[-1]

  ev = ev0 + ev_tv
  tv_weight = ev_tv / ev

  return {"enterprise_value": ev, "ev0": ev0, "ev_tv": ev_tv, "tv_weight": tv_weight}