import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import gmdh
import os

<<<<<<< HEAD
def evaluate_model_improved(y_true_orig, y_pred_scaled, data):
    scaler_y = data['scaler_y']
    
    # 1. Inverse transform from StandardScaler to Log scale
    y_pred_log = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # 2. Inverse transform from Log scale to Original scale
    # Since we used np.log1p, we use np.expm1
    y_pred_orig = np.expm1(y_pred_log)
    
    # Ensure no negative predictions (though log1p/expm1 should handle this)
    y_pred_orig = np.maximum(y_pred_orig, 0)
    
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2 = r2_score(y_true_orig, y_pred_orig)
    
    return mae, rmse, r2, y_pred_orig

def run_improved_mlp(data):
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    y_test_orig = data['y_test_orig']
    
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001],
        'learning_rate_init': [0.001],
=======
def evaluate_model(y_true_log, y_pred_log):
    # Inverse transform from log1p
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return mae, rmse, r2

def run_baseline_mlp(data):
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01],
>>>>>>> e3c5121 (Update with improved data pipeline and re-evaluated models)
        'max_iter': [500]
    }
    
    mlp = MLPRegressor(random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_mlp = grid_search.best_estimator_
<<<<<<< HEAD
    y_pred_scaled = best_mlp.predict(X_test)
    
    mae, rmse, r2, y_pred_orig = evaluate_model_improved(y_test_orig, y_pred_scaled, data)
    
    print(f"Improved MLP - Best Params: {grid_search.best_params_}")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    return {
        'model_name': 'Improved MLP (GridSearch)',
=======
    y_pred = best_mlp.predict(X_test)
    
    mae, rmse, r2 = evaluate_model(y_test, y_pred)
    
    print(f"Improved Baseline MLP - Best Params: {grid_search.best_params_}")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    return {
        'model_name': 'Improved Baseline MLP (GridSearch)',
>>>>>>> e3c5121 (Update with improved data pipeline and re-evaluated models)
        'best_params': grid_search.best_params_,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
<<<<<<< HEAD
        'y_pred_orig': y_pred_orig
    }

def run_improved_gmdh(data):
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    y_test_orig = data['y_test_orig']
=======
        'y_pred': y_pred
    }

def run_baseline_gmdh(data):
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
>>>>>>> e3c5121 (Update with improved data pipeline and re-evaluated models)
    
    model = gmdh.Combi()
    model.fit(X_train, y_train)
    
<<<<<<< HEAD
    y_pred_scaled = model.predict(X_test)
    mae, rmse, r2, y_pred_orig = evaluate_model_improved(y_test_orig, y_pred_scaled, data)
    
    print(f"Improved GMDH - Default Combi")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    return {
        'model_name': 'Improved GMDH (GridSearch)',
=======
    y_pred = model.predict(X_test)
    mae, rmse, r2 = evaluate_model(y_test, y_pred)
    
    print(f"Improved Baseline GMDH - Default Combi")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    return {
        'model_name': 'Improved Baseline GMDH (GridSearch)',
>>>>>>> e3c5121 (Update with improved data pipeline and re-evaluated models)
        'best_params': 'Default Combi',
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
<<<<<<< HEAD
        'y_pred_orig': y_pred_orig
    }

if __name__ == "__main__":
    if not os.path.exists('processed_data/data_improved.pkl'):
        print("Error: data_improved.pkl not found. Run preprocess_improved.py first.")
    else:
        data = joblib.load('processed_data/data_improved.pkl')
        results = []
        results.append(run_improved_mlp(data))
        # results.append(run_improved_gmdh(data))
        
        os.makedirs('results', exist_ok=True)
        joblib.dump(results, 'results/improved_results.pkl')
        
        # Comparison with baseline
        if os.path.exists('results/baseline_results.pkl'):
            baseline_results = joblib.load('results/baseline_results.pkl')
            print("\n--- Performance Comparison ---")
            for i, res in enumerate(results):
                base = baseline_results[i]
                print(f"Model: {res['model_name']} vs {base['model_name']}")
                print(f"  MAE Improvement: {base['mae'] - res['mae']:.4f}")
                print(f"  RMSE Improvement: {base['rmse'] - res['rmse']:.4f}")
                print(f"  R2 Improvement: {res['r2'] - base['r2']:.4f}")
=======
        'y_pred': y_pred
    }

if __name__ == "__main__":
    data = joblib.load('processed_data_improved/data.pkl')
    results = []
    results.append(run_baseline_mlp(data))
    results.append(run_baseline_gmdh(data))
    
    os.makedirs('results_improved', exist_ok=True)
    joblib.dump(results, 'results_improved/baseline_results.pkl')
>>>>>>> e3c5121 (Update with improved data pipeline and re-evaluated models)
