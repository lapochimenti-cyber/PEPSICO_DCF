import numpy as np
import pytest
from pathlib import Path
import yaml
from src.fcff.model import fcff, convergence_path, growth_path, build_params


config_path = "config/config.yaml"
base_dir = Path(__file__).resolve().parent.parent
with open(base_dir / config_path, "r") as f:
    cfg = yaml.safe_load(f)


PARAMS_V0 = build_params(cfg["fcff_parameters"])
                    
def example_params():
    """Invented numbers of the D02 example (growth years)."""
    return {"margine": [0.15] * 10, "aliquota": [0.20] * 10,
            "da": [0.04] * 5, "capex": [0.05] * 5, "nwc": [0.10] * 5,
            "rr_inf": 0.10, "k": 0.5}


def example_paths():
    return np.array([[104, 108, 112, 116, 120, 124, 128, 132, 136, 140],
                    [102, 104, 106, 108, 110, 112, 114, 116, 118, 120]],
                    dtype=float)


def test_d02_example_table():
    nopat, reinv, flow = fcff(example_paths(), 100.0, example_params())
    expected = np.array([
        # NOPAT, Reinv, FCFF for years 1-3, path 1 then path 2
        [[12.48, 1.44, 11.04], [12.96, 1.48, 11.48], [13.44, 1.52, 11.92]],
        [[12.24, 1.22, 11.02], [12.48, 1.24, 11.24], [12.72, 1.26, 11.46]],
    ])
    got = np.stack([nopat[:, :3], reinv[:, :3], flow[:, :3]], axis=-1)
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_identity_fcff_equals_nopat_minus_reinvestment():
    nopat, reinv, flow = fcff(example_paths(), 100.0, example_params())
    np.testing.assert_allclose(flow, nopat - reinv, atol=1e-12)


def test_transition_follows_arm_recursion_from_each_path_rr5():
    nopat, reinv, _ = fcff(example_paths(), 100.0, example_params())
    rr = reinv / nopat
    # path 1 grows faster than path 2: different starting rates
    assert rr[0, 4] != pytest.approx(rr[1, 4])
    for t in range(5, 10):
        expected = rr[:, t - 1] + 0.5 * (0.10 - rr[:, t - 1])
        np.testing.assert_allclose(rr[:, t], expected, atol=1e-12)


def test_year_10_keeps_1_32_of_the_gap_on_every_path():
    paths = 100.0 * np.cumprod(np.full((3, 10), [[1.00], [1.04], [1.10]]),
                            axis=1)
    nopat, reinv, _ = fcff(paths, 100.0, example_params())
    rr5, rr10 = reinv[:, 4] / nopat[:, 4], reinv[:, 9] / nopat[:, 9]
    np.testing.assert_allclose(rr10 - 0.10, (rr5 - 0.10) / 32, atol=1e-12)


def test_k_equal_1_reaches_rr_inf_at_year_6():
    p = dict(example_params(), k=1.0)
    nopat, reinv, _ = fcff(example_paths(), 100.0, p)
    np.testing.assert_allclose(reinv[:, 5:] / nopat[:, 5:], 0.10, atol=1e-12)


def test_output_shape_matches_input():
    paths = np.full((7, 10), 95.0)
    for out in fcff(paths, 93.925, PARAMS_V0):
        assert out.shape == (7, 10)


@pytest.mark.parametrize("bad_paths", [
    np.zeros((2, 10)),
    -np.ones((2, 10)),
    np.full((2, 10), np.nan),
    np.ones(10),
    np.ones((2, 9)),
])
def test_invalid_revenues_raise(bad_paths):
    with pytest.raises(ValueError):
        fcff(bad_paths, 100.0, example_params())


@pytest.mark.parametrize("base", [0.0, -1.0, float("nan")])
def test_invalid_base_revenue_raises(base):
    with pytest.raises(ValueError):
        fcff(example_paths(), base, example_params())


@pytest.mark.parametrize("tax", [-0.01, 1.01])
def test_tax_outside_unit_interval_raises(tax):
    p = example_params()
    p["aliquota"] = [tax] * 10
    with pytest.raises(ValueError):
        fcff(example_paths(), 100.0, p)


@pytest.mark.parametrize("name", ["margine", "aliquota", "da", "capex",
                                "nwc", "rr_inf", "k"])
def test_missing_parameter_raises(name):
    p = example_params()
    del p[name]
    with pytest.raises(ValueError):
        fcff(example_paths(), 100.0, p)


@pytest.mark.parametrize("name", ["margine", "da", "nwc"])
def test_wrong_length_raises(name):
    p = example_params()
    p[name] = list(p[name]) + [0.1]
    with pytest.raises(ValueError):
        fcff(example_paths(), 100.0, p)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), [0.1, 0.1]])
def test_invalid_rr_inf_raises(bad):
    p = example_params()
    p["rr_inf"] = bad
    with pytest.raises(ValueError):
        fcff(example_paths(), 100.0, p)


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, float("nan")])
def test_invalid_k_raises(bad):
    p = example_params()
    p["k"] = bad
    with pytest.raises(ValueError):
        fcff(example_paths(), 100.0, p)


def test_old_reinv_rate_key_alone_raises():
    p = example_params()
    del p["rr_inf"]
    p["reinv_rate"] = [0.10] * 5
    with pytest.raises(ValueError, match="rr_inf"):
        fcff(example_paths(), 100.0, p)


def test_non_positive_year5_nopat_raises():
    p = example_params()
    p["margine"] = [0.15] * 4 + [0.0] + [0.15] * 5
    with pytest.raises(ValueError):
        fcff(example_paths(), 100.0, p)


def test_negative_fcff_is_not_clipped():
    p = example_params()
    p["capex"] = [0.50] * 5
    _, _, flow = fcff(example_paths(), 100.0, p)
    assert np.all(flow[:, :5] < 0)


def test_convergence_path_halves_the_gap():
    np.testing.assert_allclose(convergence_path(0.10, 0.20),
                            [0.15, 0.175, 0.1875, 0.19375, 0.196875])
    np.testing.assert_allclose(convergence_path(0.10, 0.20, k=1.0), [0.20] * 5)
    with pytest.raises(ValueError):
        convergence_path(0.10, 0.20, k=0.0)


def test_growth_path_matches_g1_scenarios():
    """Base scenario of PEP_scenari.xlsx: 2031-35 growth and revenue 2035."""
    w_na = 55.725 / 93.925
    g_base = 0.5 * 0.0386 + 0.5 * (w_na * 0.01 + (1 - w_na) * 0.06)
    g = growth_path([g_base] * 5, 0.025)
    np.testing.assert_allclose(g[5:], [0.0297, 0.0274, 0.0262, 0.0256, 0.0253],
                            atol=5e-5)
    assert 93.925 * np.prod(1 + g) == pytest.approx(127.015, abs=0.02)
    with pytest.raises(ValueError):
        growth_path([0.03] * 4, 0.025)


def test_v0_params_are_labelled_provisional():
    assert PARAMS_V0["status"] == "provvisory"
    assert PARAMS_V0["k"] == cfg["fcff_parameters"]["k"] == 0.5


def test_v0_year_10_is_1_32_of_the_gap_from_maturity():
    a = cfg["fcff_parameters"]
    m5, tau5 = a["m_growth"][-1], a["tau_growth"][-1]
    assert PARAMS_V0["margine"][9] == pytest.approx(
        a["m_inf"] + (m5 - a["m_inf"]) / 32)
    assert PARAMS_V0["aliquota"][9] == pytest.approx(
        a["tau_inf"] + (tau5 - a["tau_inf"]) / 32)
    assert PARAMS_V0["rr_inf"] == pytest.approx(a["g_inf"] / a["roic_inf"])


def test_v0_deterministic_reference_path():
    """sigma = 0 path: 4% for years 1-5, then Arm recursion (k = 0.5) toward
    g_inf = 2.5%. Values computed independently with a plain-Python loop."""
    a = cfg["fcff_parameters"]
    R = cfg["revenues"]["r0"] * np.cumprod(1 + growth_path([0.04] * 5, a["g_inf"]))
    nopat, reinv, flow = fcff(R[None, :], cfg["revenues"]["r0"], PARAMS_V0)
    expected_fcff = [10.3319, 10.7451, 11.1749, 11.6219, 12.0868,
                    12.4937, 12.8583, 13.2062, 13.5498, 13.8954]
    np.testing.assert_allclose(flow[0], expected_fcff, atol=1e-3)
    assert nopat[0, 0] == pytest.approx(12.1690, abs=1e-3)
    assert reinv[0, 9] == pytest.approx(1.9998, abs=1e-3)
    assert reinv[0, 4] / nopat[0, 4] == pytest.approx(0.1510, abs=1e-4)
    assert reinv[0, 9] / nopat[0, 9] == pytest.approx(0.1258, abs=1e-4)


def test_build_params_rejects_g_above_roic():
    a = dict(cfg["fcff_parameters"], g_inf=0.25)
    with pytest.raises(ValueError):
        build_params(a)