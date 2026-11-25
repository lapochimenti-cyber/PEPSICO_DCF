import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml

from src.revenues.model import estimate_sigma

BASE_DIR = Path(__file__).resolve().parents[2]


def calibrate_sigma(revenue_config: dict) -> dict:
    path = BASE_DIR / revenue_config['historical_csv']
    column = revenue_config['historical_column']
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter=';'))
    try:
        years = np.asarray([int(row['Anno']) for row in rows])
        revenues = np.asarray([float(row[column]) for row in rows])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Invalid historical revenue columns or values') from exc
    if len(years) < 3 or not np.all(np.diff(years) == 1):
        raise ValueError('Historical years must be consecutive, unique and ascending')
    sigma = estimate_sigma(revenues)
    if not np.isclose(revenues[-1], revenue_config['r0'], rtol=0, atol=1e-9):
        raise ValueError('Base revenue must match the last historical observation')
    return {
        'sigma': sigma,
        'status': revenue_config['sigma_status'],
        'source': revenue_config['historical_csv'],
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'column': column,
        'units': 'annual decimal volatility; revenues in billions USD',
        'start_year': int(years[0]), 'end_year': int(years[-1]),
        'revenue_observations': len(revenues), 'log_return_observations': len(revenues)-1,
        'ddof': 1, 'method': 'std(log(R[t] / R[t-1]), ddof=1)',
        'reference': 'Guida Valuation PEP, B7',
        'log_returns': np.log(revenues[1:] / revenues[:-1]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--output', help='Optional JSON path relative to the project root')
    args = parser.parse_args()
    with (BASE_DIR / args.config).open(encoding='utf-8') as stream:
        cfg = yaml.safe_load(stream)
    report = json.dumps(calibrate_sigma(cfg['revenues']), indent=2) + '\n'
    if args.output:
        dest = BASE_DIR / args.output
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(report, encoding='utf-8')
    print(report, end='')


if __name__ == '__main__':
    main()