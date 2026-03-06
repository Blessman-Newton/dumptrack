import joblib
import os

from baseline_models import run_baseline_gmdh, run_baseline_mlp
from preprocess import preprocess_data


if __name__ == '__main__':
    data = preprocess_data('cleaned_data.csv')
    results = []

    print('Running Baseline MLP...')
    results.append(run_baseline_mlp(data))

    print('Running Baseline GMDH...')
    results.append(run_baseline_gmdh(data))

    os.makedirs('results', exist_ok=True)
    joblib.dump(results, 'results/baseline_results.pkl')
    print('Baseline results saved.')
