"""Backward-compatible entrypoint for improved baseline models."""

from baseline_models import run_baseline_gmdh, run_baseline_mlp
import joblib
import os


if __name__ == '__main__':
    data = joblib.load('processed_data/data.pkl')
    results = [run_baseline_mlp(data), run_baseline_gmdh(data)]

    os.makedirs('results_improved', exist_ok=True)
    joblib.dump(results, 'results_improved/baseline_results.pkl')
