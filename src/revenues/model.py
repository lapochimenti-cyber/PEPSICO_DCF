import numpy as np
import pandas as pd
from pathlib import Path


def estimate_sigma (historical_revenues: np.array) -> float:

  if np.any(historical_revenues <= 0):
    raise ValueError("Historical revenues must be positive")   

  log_returns = np.log(historical_revenues[1:] / historical_revenues[:-1])

  # if we want to use as stated in the guide the Bessel approximation for the std we can write
  
  return float(np.std(log_returns, ddof = 1))


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

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    csv_path = BASE_DIR / "data" / "D04_ricavi_v1_G1.csv"

    # read the CSV file specifiying to the compiler that the separator used is the ';'
    df = pd.read_csv(csv_path, sep=";")

    df = df.sort_values("Anno").reset_index(drop=True)
    ricavi_storici = df["Ricavi Corretti ($B)"].to_numpy(dtype=float)

    sigma_hat = estimate_sigma(ricavi_storici)
    r0 = float(ricavi_storici[-1])
    anno_base = int(df["Anno"].iloc[-1])

    print("=" * 55)
    print("CALIBRAZIONE RICAVI DA D04 (G1 -> G4)")
    print("=" * 55)
    print(f"Anni analizzati:            {df['Anno'].iloc[0]} - {anno_base} ({len(ricavi_storici)-1} variazioni YoY)")
    print(f"Ricavo anno base r0 ({anno_base}): {r0:.3f} ($B)")
    print(f"Sigma stimato (ddof=1):     {sigma_hat:.6f} ({sigma_hat*100:.2f}%)")
    print("=" * 55)

    
