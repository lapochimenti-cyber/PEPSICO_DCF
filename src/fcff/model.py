# Hey there G2, it's your turn to write some code now! Good work...
import numpy as np

N_YEARS = 10
N_GROWTH = 5
K_GUIDE = 0.5   # convergence rate of the course guide (Arm paper, appendix B)

LENGTHS = {"margine": 10, "aliquota": 10, "da": 5, "capex": 5, "nwc": 5}


def fcff(ricavi, ricavo_base, params):
    """Return (nopat, reinvest, fcff), each np.ndarray of shape (N, 10).

    ricavi       (N, 10) simulated revenues, bn USD, column 0 = FY2026
    ricavo_base  FY2025 revenue, bn USD (from the shared config file)
    params       dict with the keys in LENGTHS plus the scalars rr_inf and k;
                 extra keys are ignored

    In years 6..10 each path starts from its own year-5 reinvestment rate
    rr_5 = Reinv_5 / NOPAT_5 and converges to rr_inf with rate k, so the
    transition has no jump at year 6.

    Raises ValueError on non-positive revenues, tax rates outside [0, 1],
    k outside (0, 1], non-positive year-5 NOPAT, missing parameters or wrong
    lengths. No clipping: negative FCFF is returned as is.
    """
    R = np.asarray(ricavi, dtype=float)
    if R.ndim != 2 or R.shape[1] != N_YEARS:
        raise ValueError("ricavi must have shape (N, 10)")
    if not np.isfinite(ricavo_base) or ricavo_base <= 0:
        raise ValueError("ricavo_base must be a positive number")
    if not np.all(np.isfinite(R)) or np.any(R <= 0):
        raise ValueError("ricavi must be finite and strictly positive")

    p = {}
    for name, n in LENGTHS.items():
        if name not in params:
            raise ValueError(f"missing parameter: {name}")
        p[name] = np.asarray(params[name], dtype=float)
        if p[name].shape != (n,):
            raise ValueError(f"{name} must have length {n}")
        if not np.all(np.isfinite(p[name])):
            raise ValueError(f"{name} contains NaN or inf")
    if np.any((p["aliquota"] < 0) | (p["aliquota"] > 1)):
        raise ValueError("aliquota outside [0, 1]")
    for name in ("rr_inf", "k"):
        if name not in params:
            raise ValueError(f"missing parameter: {name}")
        value = np.asarray(params[name], dtype=float)
        if value.shape != () or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite scalar")
        p[name] = float(value)
    check_k(p["k"])
           
    # rr_5 divides by NOPAT_5, which must be positive on every path
    if p["margine"][N_GROWTH - 1] <= 0 or p["aliquota"][N_GROWTH - 1] >= 1:
        raise ValueError("year-5 margine must be > 0 and aliquota < 1")

    nopat = R * p["margine"] * (1.0 - p["aliquota"])
    R_prev = np.column_stack([np.full(R.shape[0], float(ricavo_base)), R[:, :-1]])

    reinvest = np.empty_like(R)
    g = slice(0, N_GROWTH)
    reinvest[:, g] = (R[:, g] * (p["capex"] - p["da"]) + (R[:, g] - R_prev[:, g]) * p["nwc"])

    rr5 = reinvest[:, N_GROWTH - 1] / nopat[:, N_GROWTH - 1]
    rr = convergence_path(rr5[:, None], p["rr_inf"], p["k"])
    reinvest[:, N_GROWTH:] = rr * nopat[:, N_GROWTH:]

    return nopat, reinvest, nopat - reinvest


def check_k(k):
    if not 0 < k <= 1:
        raise ValueError("k must be in (0, 1]")


def convergence_path(x_start, x_inf, k=K_GUIDE, n=N_GROWTH):
    """Values for years 6..10 from the recursion x_t = x_{t-1} + k (x_inf - x_{t-1}).

    Closed form: x_{5+j} = x_inf + (1 - k)**j (x_start - x_inf), j = 1..n.
    With k = 0.5 year 10 keeps 1/32 of the initial gap; the maturity value
    x_inf applies from year 11 (terminal value). x_start may be a column
    (N, 1): the result is then (N, n), one row per path.
    """
    check_k(k)
    decay = (1.0 - k) ** np.arange(1, n + 1)
    return x_inf + decay * (x_start - x_inf)


def growth_path(growth_explicit, g_inf, k=K_GUIDE):
    """10 yearly growth rates: years 1..5 as given, years 6..10 converging to
    g_inf with the same recursion. Used for the bear / base / growth
    scenarios (one call per scenario); in the Monte Carlo it is the drift of
    each scenario.
    """
    g = np.asarray(growth_explicit, dtype=float)
    if g.shape != (N_GROWTH,):
        raise ValueError("growth_explicit must have length 5")
    return np.concatenate([g, convergence_path(g[-1], g_inf, k)])


def build_params(a):
    #Build the fcff params dict from an assumptions dict.
    if not 0 < a["g_inf"] < a["roic_inf"]:
        raise ValueError("need 0 < g_inf < roic_inf")

    k = a["k"]
    m = np.asarray(a["m_growth"], dtype=float)
    tau = np.asarray(a["tau_growth"], dtype=float)

    return {
        "status": a["status"],
        "margine": np.concatenate([m, convergence_path(m[-1], a["m_inf"], k)]),
        "aliquota": np.concatenate([tau, convergence_path(tau[-1], a["tau_inf"], k)]),
        "da": np.asarray(a["da"], dtype=float),
        "capex": np.asarray(a["capex"], dtype=float),
        "nwc": np.asarray(a["nwc"], dtype=float),
        "rr_inf": a["g_inf"] / a["roic_inf"],
        "k": k,
    }

