import numpy as np

class EquityBridge:
    def __init__(self, bridge_inputs, diluted_shares, provisional=()):
        # Initializing bridge with a dictionary of values in USD billions
        self.bridge_inputs = bridge_inputs
        self.diluted_shares = diluted_shares
        self.provisional = list(provisional)  # names of items still provisional

        self.bridge_total = sum(self.bridge_inputs.values())

    def assert_final(self):
        # Final run: no provisional item allowed
        if self.provisional:
            raise ValueError(f"Provisional bridge items: {', '.join(self.provisional)}")

    def calculate_value_share(self, ev_oper):
        equity_value = ev_oper + self.bridge_total
        price_per_share = equity_value / self.diluted_shares

        breakdown = {"Operating Enterprise Value": ev_oper}
        breakdown.update(self.bridge_inputs)
        breakdown.update({
            "Equity value": equity_value,
            "Diluted shares (bn)": self.diluted_shares,
            "Value per share (USD)": price_per_share
        })
        return equity_value, price_per_share, breakdown


# --- FUNZIONE DI INTEGRAZIONE PER LA PIPELINE MONTE CARLO (G4) ---
def compute_equity_and_price(ev: np.ndarray, bridge_params: dict, is_final_run: bool = False) -> dict:
    """
    Usa la classe EquityBridge di G3 per passare da EV a Equity Value e Prezzo per Azione.
    """
    shares = float(bridge_params["diluted_shares"])
    if shares <= 0:
        raise ValueError("Il numero di azioni diluite deve essere strettamente positivo.")

    # Se nel config.yaml abbiamo il dizionario dettagliato di G3 ('items'), usiamo quello;
    # altrimenti supportiamo anche il formato sintetico della Settimana 1.
    if "items" in bridge_params:
        bridge_inputs = {k: float(v) for k, v in bridge_params["items"].items()}
        provisional = bridge_params.get("provisional_items", [])
    else:
        bridge_inputs = {
            "Cash": float(bridge_params.get("cash", 0.0)),
            "Non-operating assets": float(bridge_params.get("non_operating_assets", 0.0)),
            "Total debt": -float(bridge_params.get("total_debt", 0.0)),
            "Minorities and other": -float(bridge_params.get("minorities_and_other", 0.0)),
        }
        provisional = bridge_params.get("provisional_items", [])

    bridge = EquityBridge(bridge_inputs=bridge_inputs, diluted_shares=shares, provisional=provisional)

    if is_final_run:
        bridge.assert_final()

    equity_val, share_price, breakdown = bridge.calculate_value_share(ev)

    return {
        "equity_value": equity_val,
        "share_price": share_price,
        "bridge_total": bridge.bridge_total,
        "negative_equity_count": int(np.sum(equity_val < 0)),
        "breakdown": breakdown
    }


if __name__ == "__main__":
    # D03 (accepted by G2): operating leases (3.846) and JVs (2.215) stay in the flows, not here
    tax = 0.21  # H10: same rate as the WACC
    data_bridge = {
        "Cash and cash equivalents": 10.675,                            # 10-Q p. 6, 2026-09-05
        "Short-term investments": 0.488,                                # 10-Q p. 6
        "Tropicana Brands Group (39%)": 0.0,                            # 10-K p. 96, fully written down
        "Celsius convertible preferred (fair value)": 1.482,            # 10-Q p. 22
        "Short-term debt obligations": -9.223,                          # 10-Q p. 6
        "Long-term debt obligations": -42.658,                          # 10-Q p. 6
        "Pension and retiree medical deficits, after tax": -(0.233 + 0.468) * (1 - tax),  # 10-K p. 86
        "Dividends payable": -(1.967 + 6.004 - 5.935),
        "Non-controlling interests": -0.195                             # 10-Q p. 6; H13
    }
    diluted_shares = 1.366  # 10-Q cover and p. 26; H12
    provisional = []

    bridge = EquityBridge(data_bridge, diluted_shares, provisional)
    print(f'total sum of the bridge: {bridge.bridge_total:.3f} USD bn')  # expected -42.021

    # Test
    ev_test = 250.0
    eq_val, p_share, details = bridge.calculate_value_share(ev_test)

    print(f'\n Results with EV_oper = {ev_test} USD bn: \n')
    print(f'Equity Value: {eq_val:.3f} USD bn\n')
    print(f'Price per share (P): {p_share:.2f} USD')
