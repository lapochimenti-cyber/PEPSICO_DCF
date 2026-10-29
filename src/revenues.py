import numpy as np

def estimate_sigma (historical_revenues: np.array) -> float:

  if np.any(historical_revenues <= 0):
    raise ValueError("Historical revenues must be positive")   

  log_returns = np.log(historical_revenues[1:] / historical_revenues[:-1])

  # if we want to use as stated in the guide the Bessel approximation for the std we can write
  # return float(np.std(log_returns, ddof = 1))

  return float(np.std(log_returns))

def simulate_revenues(r0: float, g_path: np.ndarray, sigma: float, n_paths: int, seed: int) -> np.ndarray:

  # we have to simulate n_paths paths in a time frame givcen by H:
  # R_t = R_{t-1} * exp(log(1 + g_t) - 0.5 * sigma^2 + sigma * Z_t)

  if r0 <= 0:
    raise ValueError (f"r0 must be positive, while now r0 = {r0}")
  if np.any(g_path <= -1.0):
    raise ValueError("the groth rate must be grater than -1.0")
  if sigma < 0:
    raise ValueError(f"sigma must be positive, while now sigma = {sigma}")

  horizon = len(g_path)
  
  rng = np.random.default_rng(seed)                 
  z = rng.standard_normal(size=(n_paths, horizon))

  drift = np.log(1.0 + g_path) - 0.5 * sigma**2
  log_growth = drift + sigma * z

  cumulative_log_growth = np.cumsum(log_growth, axis=1)
  revenues_matrix = np.exp(cumulative_log_growth) * r0

  return revenues_matrix
