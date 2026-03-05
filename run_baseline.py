import joblib
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from preprocess import preprocess_data
from baseline_models import run_baseline_mlp, run_baseline_gmdh

if __name__ == "__main__":
    data = preprocess_data('cleaned_data.csv')
    results = []
    print("Running Baseline MLP...")
    results.append(run_baseline_mlp(data))
    # Skip GMDH if it causes segmentation fault
    # print("Running Baseline GMDH...")
    # results.append(run_baseline_gmdh(data))
    
    os.makedirs('results', exist_ok=True)
    joblib.dump(results, 'results/baseline_results.pkl')
    print("Baseline results saved.")
