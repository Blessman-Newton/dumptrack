import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import gmdh
import os

def evaluate_model(y_true, y_pred, scaler_y):
    # Inverse transform to get original scale
    y_true_orig = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2 = r2_score(y_true_orig, y_pred_orig)
    
    return mae, rmse, r2

def run_baseline_mlp(data):
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [500]
    }
    
    mlp = MLPRegressor(random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_mlp = grid_search.best_estimator_
    y_pred = best_mlp.predict(X_test)
    
    mae, rmse, r2 = evaluate_model(y_test, y_pred, data['scaler_y'])
    
    print(f"Baseline MLP - Best Params: {grid_search.best_params_}")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    return {
        'model_name': 'Baseline MLP (GridSearch)',
        'best_params': grid_search.best_params_,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_pred': y_pred
    }

def run_baseline_gmdh(data):
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    
    # GMDH implementation using the gmdh library
    # The gmdh library has different API. Let's use Combi with default settings or simple tuning.
    # Based on documentation, Combi().fit(X, y) is the standard way.
    
    model = gmdh.Combi()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae, rmse, r2 = evaluate_model(y_test, y_pred, data['scaler_y'])
    
    print(f"Baseline GMDH - Default Combi")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    return {
        'model_name': 'Baseline GMDH (GridSearch)',
        'best_params': 'Default Combi',
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_pred': y_pred
    }

if __name__ == "__main__":
    data = joblib.load('processed_data/data.pkl')
    results = []
    results.append(run_baseline_mlp(data))
    results.append(run_baseline_gmdh(data))
    
    os.makedirs('results', exist_ok=True)
    joblib.dump(results, 'results/baseline_results.pkl')
