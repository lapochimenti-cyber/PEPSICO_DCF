# Hey there G2, it's your turn to write some code now! Good work...
import numpy as np

N_YEARS = 10
N_GROWTH = 5

LENGTHS = {"margine": 10, "aliquota": 10, "da": 5,
           "capex": 5, "nwc": 5, "reinv_rate": 5}


def fcff(ricavi, ricavo_base, params):
    """Return (nopat, reinvest, fcff), each np.ndarray of shape (N, 10).

    ricavi       (N, 10) simulated revenues, bn USD, column 0 = FY2026
    ricavo_base  FY2025 revenue, bn USD (from the shared config file)
    params       dict with the keys in LENGTHS; extra keys are ignored

    Raises ValueError on non-positive revenues, tax rates outside [0, 1],
    missing parameters or wrong lengths. No clipping: negative FCFF is
    returned as is.
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

    nopat = R * p["margine"] * (1.0 - p["aliquota"])
    R_prev = np.column_stack([np.full(R.shape[0], float(ricavo_base)),
                              R[:, :-1]])

    reinvest = np.empty_like(R)
    g = slice(0, N_GROWTH)
    reinvest[:, g] = (R[:, g] * (p["capex"] - p["da"])
                      + (R[:, g] - R_prev[:, g]) * p["nwc"])
    reinvest[:, N_GROWTH:] = p["reinv_rate"] * nopat[:, N_GROWTH:]

    return nopat, reinvest, nopat - reinvest


def linear_path(x_start, x_inf, n=N_GROWTH):
    """Values for years 6..10 moving linearly from x_start to x_inf."""
    w = np.arange(1, n + 1) / n
    return x_start + w * (x_inf - x_start)


def build_params(a):
    """Build the fcff params dict from an assumptions dict.

    The transition starts from the year-5 reinvestment rate of the
    deterministic path R_t = R_0 * prod(1 + growth) (option A of D02):
    one rr_5 shared by all paths.
    """
    if not 0 < a["g_inf"] < a["roic_inf"]:
        raise ValueError("need 0 < g_inf < roic_inf")

    m = np.asarray(a["m_growth"], dtype=float)
    tau = np.asarray(a["tau_growth"], dtype=float)
    da = np.asarray(a["da"], dtype=float)
    capex = np.asarray(a["capex"], dtype=float)
    nwc = np.asarray(a["nwc"], dtype=float)
    growth = np.asarray(a["growth"], dtype=float)

    R = a["ricavo_base"] * np.cumprod(1.0 + growth)
    nopat5 = R[-1] * m[-1] * (1.0 - tau[-1])
    reinv5 = R[-1] * (capex[-1] - da[-1]) + (R[-1] - R[-2]) * nwc[-1]
    rr5 = reinv5 / nopat5

    return {
        "status": a["status"],
        "margine": np.concatenate([m, linear_path(m[-1], a["m_inf"])]),
        "aliquota": np.concatenate([tau, linear_path(tau[-1], a["tau_inf"])]),
        "da": da,
        "capex": capex,
        "nwc": nwc,
        "reinv_rate": linear_path(rr5, a["g_inf"] / a["roic_inf"]),
    }


# Every value below is PROVISIONAL except ricavo_base. Page numbers are the
# printed pages of the FY2025 10-K (PDF page = printed + 3).
ASSUMPTIONS_V0 = {
    "status": "provisional",
    "ricavo_base": 93.925,   # real: net revenue FY2025, 10-K p. 57
    "growth": [0.04] * 5,    # placeholder until G1 scenarios (D05)
    "m_growth": [0.159] * 5,  # core op. margin FY2025, 14.912/93.925, p. 42; until G1 D10
    "m_inf": 0.155,          # placeholder until D14
    "tau_growth": [0.19] * 5,  # effective rate FY2025, 1.949/10.244, p. 57
    "tau_inf": 0.20,         # placeholder until H10 with G3
    "da": [0.035] * 5,       # D&A / revenue, FY2016-25 average
    "capex": [0.055] * 5,    # capex / revenue, FY2016-25 average
    "nwc": [-0.05] * 5,      # operating NWC / revenue FY2025 (-4.668/93.925)
    "g_inf": 0.025,          # placeholder until G3
    "roic_inf": 0.20,        # placeholder until D14
}

PARAMS_V0 = build_params(ASSUMPTIONS_V0)