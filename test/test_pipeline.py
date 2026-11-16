import pytest
import numpy as np
from pathlib import Path
from src.pipeline import run_pipeline, load_g_path

def test_sigma_zero_reproduces_deterministic_dcf():
    # Sez. 14.5 test: for Sigma = 0 it needs to replicate the DFC model with null variance, all 10000 paths need to be equal to the deterministic one
    res = run_pipeline(config_path = "config/config.yaml",  is_final_run = False, override_sigma=0.0)
    prices = res["prices_array"]
    
    assert np.allclose(prices, prices[0], atol=1e-12)
    assert np.isclose(res["price_mean"], res["price_median"], atol=1e-12)

def test_monte_carlo_revenue_mean_matches_expected_trajectory():
    # Sez. 14.5: the synthetic revenues' mean needs to follow the drift in three std top
    res = run_pipeline(config_path = "config/config.yaml",  is_final_run = False, override_sigma=0.15)
    revs = res["revenues_matrix"] # (n_paths, H)

    # please insert the right values put in the yaml file
    r0 = 93.925
    g_path = np.array([
    0.0345,       # 2026 (Anno 1)
    0.0345,       # 2027 (Anno 2)
    0.0345,       # 2028 (Anno 3)
    0.0345,       # 2029 (Anno 4)
    0.0345,       # 2030 (Anno 5)
    0.02975,      # 2031 (Anno 6)
    0.027375,     # 2032 (Anno 7)
    0.0261875,    # 2033 (Anno 8)
    0.02559375,   # 2034 (Anno 9)
    0.025296875,  # 2035 (Anno 10)
    ])
    expected_revs = r0 * np.cumprod(1.0 + g_path)
    
    sample_means = np.mean(revs, axis=0)
    mc_std_errors = np.std(revs, axis=0, ddof=1) / np.sqrt(revs.shape[0])
    
    assert np.all(np.abs(sample_means - expected_revs) < 3.0 * mc_std_errors)

def test_final_run_blocks_provisional_data():
    # the final run needs to stop if the data are not the real ones
    with pytest.raises(RuntimeError, match="Pipeline block: all data need to be in status 'real'!"):
        run_pipeline(config_path = "config/config.yaml",  is_final_run=True, override_sigma = None)
