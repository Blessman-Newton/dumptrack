import joblib
import os

def print_res(file_path, label):
    if os.path.exists(file_path):
        results = joblib.load(file_path)
        print(f"\n--- {label} ---")
        for res in results:
            print(f"Model: {res['model_name']}")
            print(f"  MAE: {res['mae']:.4f}")
            print(f"  RMSE: {res['rmse']:.4f}")
            print(f"  R2: {res['r2']:.4f}")
    else:
        print(f"{label} file not found.")

if __name__ == "__main__":
    print_res('results/baseline_results.pkl', 'Baseline Results')
    print_res('results/improved_results.pkl', 'Improved Results')
