# Here we go again G3, better start working...
import numpy as np

def compute_equity_and_price(ev: np.ndarray, bridge_parameters: dict, g3_functions: dict) -> dict:

  # G3 owns the bridge calculation. This is a local adapter contract to be mapped to their actual repo function once available.
  # EV and signed bridge amounts: billions of USD.
  # Diluted shares: billions of shares; returned price: USD per share.

  shares = float(bridge_parameters["diluted_shares"])

  if not np.isfinite(shares) or shares <= 0:
    raise ValueError("The number of diluted shares must be finite and positive")

  for item in bridge_parameters["items"]:
    if item["sign"] not in (-1, 1):
      raise ValueError("Each bridge item must have sign -1 or +1")

    if not np.isfinite(float(item["value"])) or float(item["value"]) < 0:
      raise ValueError("Bridge item values must be finite unsigned amounts")

  result = g3_functions["compute_equity_and_price"](ev=ev, bridge_parameters=bridge_parameters)
  equity_val = np.asarray(result["equity_value"], dtype=float)
  share_price = np.asarray(result["share_price"], dtype=float)

  if equity_val.shape != ev.shape or share_price.shape != ev.shape:
    raise ValueError("G3 must return one equity value and price per path")

  if not np.all(np.isfinite(equity_val)) or not np.all(np.isfinite(share_price)):
    raise ValueError("G3 equity values and prices must be finite")

  return {"equity_value": equity_val, "share_price": share_price,
          "negative_equity_count": int(np.sum(equity_val < 0))}
