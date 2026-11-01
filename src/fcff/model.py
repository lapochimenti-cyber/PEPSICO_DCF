# Hey there G2, it's your turn to write some code now! Good work...
import numpy as np

# Local implementation of D02 v2, option A; replace with G2's implementation when supplied.
def fcff(ricavi: np.ndarray, ricavo_base: float, params: dict) -> tuple:

  # first check
  try:
    revenues = np.asarray(ricavi, dtype=float)
    ricavo_base = float(ricavo_base)
  except (TypeError, ValueError) as exc:
    raise ValueError("Revenues and base revenue must be numeric") from exc
  if revenues.ndim != 2 or revenues.shape[1] != 10:
    raise ValueError("Revenues must have shape (N, 10)")
  if not np.all(np.isfinite(revenues)) or np.any(revenues <= 0):
    raise ValueError("All revenues must be finite and positive")
  if not np.isfinite(ricavo_base) or ricavo_base <= 0:
    raise ValueError("Base revenue must be finite and positive")

  required_inputs = {"margine": 10, "aliquota": 10, "da": 5,
                     "capex": 5, "nwc": 5, "reinv_rate": 5}

  # second check
  if not isinstance(params, dict):
    raise ValueError("params must be a dictionary")
  for k in required_inputs:
    if k not in params or params[k] is None:
      raise ValueError(f"The parameter {k} is missing")

  # it is useful to convert the previous data listed at the beginning into arrays on which we can work easierly
  arrays = {}
  for k, length in required_inputs.items():
    try:
      arrays[k] = np.asarray(params[k], dtype=float)
    except (TypeError, ValueError) as exc:
      raise ValueError(f"The parameter {k} must be numeric") from exc
    if arrays[k].shape != (length,) or not np.all(np.isfinite(arrays[k])):
      raise ValueError(f"The parameter {k} must contain {length} finite values")
  margin = arrays["margine"]
  tax = arrays["aliquota"]
  reinv_rate = arrays["reinv_rate"]

  # third check
  if np.any((tax < 0.0) | (tax > 1.0)):
    raise ValueError("The tax rate must be between 0 and 1")

  # we can now do operation on the arrays alltogheter saving time?
  nopat = revenues * margin * (1.0 - tax)
  previous_revenues = np.concatenate(
      (np.full((revenues.shape[0], 1), ricavo_base), revenues[:, :4]), axis=1)
  reinvest_first = (revenues[:, :5] * (arrays["capex"] - arrays["da"])
                    + (revenues[:, :5] - previous_revenues) * arrays["nwc"])
  reinvest_last = nopat[:, 5:] * reinv_rate
  reinvestement = np.concatenate((reinvest_first, reinvest_last), axis=1)
  flussi = nopat - reinvestement

  return nopat, reinvestement, flussi
