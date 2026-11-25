# src/rates/model.py
import pandas as pd
import numpy as np

'''
    WACC, Terminal Value, and Enterprise Value (Modulo G3 integrato da G4)
    - Calculates 3-stage WACC and convergence vector over 10 years.
    - Computes cumulative Discount Factors (DF_t).
    - Estimates Terminal Value (TV) via value driver model.
    - Computes Enterprise Value (EV) bridging FCFF flows from G2.
'''

class WaccTerminalCalculator:
    def __init__(self, price_share, beta, tax, rf=0.0496, erp=0.042, rd=0.0566,
                 diluted_shares=1.366, debt_total=(9.223 + 42.658),
                 g_inf=0.02, roic_inf=0.10, k_transition=0.5):
        # Valori di default da D08 v1 (sovrascrivibili da config.yaml)
        self.rf = rf
        self.beta = beta
        self.erp = erp
        self.rd = rd
        self.tax = tax
        self.diluted_shares = diluted_shares
        self.debt_total = debt_total
        self.price_share = price_share

        # Terminal Value parameters
        self.g_inf = g_inf
        self.roic_inf = roic_inf
        self.k_transition = k_transition

    def calculate_wacc_base(self):
        # Vector WACC for growth stage (5 years)
        D = self.debt_total
        E = self.price_share * self.diluted_shares
        re = self.rf + (self.beta * self.erp)
        wacc = (E * re + D * self.rd * (1 - self.tax)) / (D + E)
        return wacc

    def generate_wacc_vector(self, wacc_growth, wacc_maturity):
        # Vector WACC for 10 years
        wacc_vector = []

        # Years 1-5: Growth
        for _ in range(5):
            wacc_vector.append(wacc_growth)

        # Years 6-10: Transition
        w_current = wacc_growth
        for _ in range(5):
            w_current += self.k_transition * (wacc_maturity - w_current)
            wacc_vector.append(w_current)

        return wacc_vector

    def calculate_terminal_value(self, nopat_h1, w_inf, g_inf, ROIC_inf):
        # TV with value driver formula (funziona sia su scalare sia su array di 10.000 percorsi)
        if w_inf <= g_inf:
            raise ValueError("maturity WACC must be > g_inf")
        if ROIC_inf < w_inf:
            raise ValueError("terminal ROIC must not be lower than the maturity WACC")
        reinvestment_rate = g_inf / ROIC_inf
        TV = nopat_h1 * (1 - reinvestment_rate) / (w_inf - g_inf)
        return TV
    
    '''
    def calculate_discount_factors(self, wacc_vector):
        # No rounding: with sigma = 0 the run must match the deterministic DCF
        df_vector = []
        df_cumulate = 1.0
        for wacc_year in wacc_vector:
            df_cumulate = df_cumulate * (1 / (1 + wacc_year))
            df_vector.append(df_cumulate)
        return df_vector
    '''

    def calculate_discount_factors(self, wacc_vector, first_discount_period: float = 1.0):
        # No rounding: with sigma = 0 the run must match the deterministic DCF
        first_period = float(first_discount_period)
        if not np.isfinite(first_period) or first_period <= 0:
            raise ValueError("first_discount_period must be positive")

        df_vector = []
        for idx, wacc_year in enumerate(wacc_vector):
            if idx == 0:
                df_cumulate = (1.0 + wacc_year) ** (-first_period)
            else:
                df_cumulate = df_cumulate / (1.0 + wacc_year)
            df_vector.append(df_cumulate)
        return df_vector
    
    def calculate_enterprise_value(self, TV, df_vector, fcff_vector):
        # One path at a time: fcff_vector has the 10 yearly FCFF of G2
        H = 10  # total years
        ev0 = 0.0
        for i in range(H):
            ev0 += fcff_vector[i] * df_vector[i]
        ev0 += TV * df_vector[H - 1]
        return ev0


# --- FUNZIONE DI INTEGRAZIONE PER LA PIPELINE MONTE CARLO (G4) ---
def compute_enterprise_value(revenues: np.ndarray, fcff: np.ndarray, params: dict, nopat_h1: np.ndarray = None, w_inf: float = None, s_inf: float = None) -> dict:

    price_share = float(params.get("price_share", 128.34))
    beta_growth = float(params.get("beta_growth", 0.511))
    beta_maturity = float(params.get("beta_maturity", 0.55))
    tax = float(params.get("tax_marginal", 0.21))
    g_inf = float(params.get("g_terminal", 0.025))
    roic_inf = w_inf + s_inf
    m_inf = float(params.get("margin_terminal", 0.16))

    calc_growth = WaccTerminalCalculator(
        price_share=price_share, beta=beta_growth, tax=tax,
        rf=float(params.get("rf", 0.0522)), erp=float(params.get("erp", 0.042)),
        rd=float(params.get("rd", 0.0592)), diluted_shares=float(params.get("diluted_shares", 1.366)),
        debt_total=float(params.get("debt_total_wacc", 9.223 + 42.658)),
        g_inf=g_inf, roic_inf=roic_inf, k_transition=float(params.get("k_transition", 0.5))
    )
    calc_maturity = WaccTerminalCalculator(
        price_share=price_share, beta=beta_maturity, tax=tax,
        rf=calc_growth.rf, erp=calc_growth.erp, rd=calc_growth.rd,
        diluted_shares=calc_growth.diluted_shares, debt_total=calc_growth.debt_total
    )

    w_growth = calc_growth.calculate_wacc_base()
    w_maturity = calc_maturity.calculate_wacc_base()

    wacc_vector = calc_growth.generate_wacc_vector(w_growth, w_maturity)
    # La consegna YAML conserva la precisione dei tassi approvati da G3.
    if "wacc_by_year" in params:
        wacc_vector = np.asarray(params["wacc_by_year"], dtype=float)
        w_maturity = float(params["wacc_maturity"])
        if wacc_vector.shape != (10,) or not np.all(np.isfinite(wacc_vector)) or np.any(wacc_vector <= -1):
            raise ValueError("Invalid D13 WACC vector")
    df_vector = np.asarray(calc_growth.calculate_discount_factors(wacc_vector, params["first_discount_period"]), dtype=float)

    if nopat_h1 is None:
        r_h = revenues[:, -1]
        nopat_h1 = r_h * (1.0 + g_inf) * m_inf * (1.0 - float(params["tax_terminal"]))

    tv_h = calc_growth.calculate_terminal_value(nopat_h1, w_maturity, g_inf, roic_inf)

    fraction = float(params["first_year_fraction"])
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Invalid first-year fraction")
    # G2 passa esercizi interi: la quota residua si applica solo qui.
    cashflows = np.array(fcff, dtype=float, copy=True)
    cashflows[:, 0] *= fraction
    pv_explicit = np.sum(cashflows * df_vector[None, :], axis=1)
    pv_tv = tv_h * df_vector[-1]
    ev = pv_explicit + pv_tv

    return {
        "ev": ev,
        "pv_explicit": pv_explicit,
        "pv_tv": pv_tv,
        "tv_weight": pv_tv / ev,
        "wacc_vector": wacc_vector,
        "df_vector": df_vector
    }


if __name__ == "__main__":
    # Values Test: D13 v2 (expected WACC 0.0675 growth, 0.0688 maturity)
    price_share = 128.34  # close 2026-10-08
    beta = 0.511           # observed beta of PEP (G5, D20) until D09
    beta_maturity = 0.55  # G3 assumption, to be decided
    tax = 0.21

    calc = WaccTerminalCalculator(price_share, beta, tax)
    w_growth = calc.calculate_wacc_base()
    w_maturity = WaccTerminalCalculator(price_share, beta_maturity, tax).calculate_wacc_base()

    vector = calc.generate_wacc_vector(w_growth, w_maturity)
    df_factors = calc.calculate_discount_factors(vector)

    print(f'Vector WACC (10 years):', [round(w, 4) for w in vector])
    print(f"\nDiscount Factors (DF_t): {[round(d, 4) for d in df_factors]}")