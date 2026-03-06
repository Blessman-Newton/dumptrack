import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import gmdh
from mealpy.swarm_based import MFO, CSA, WOA
from mealpy import FloatVar, IntegerVar, Problem
import os

def evaluate_model(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def fitness_mlp(solution, data):
    h_size = int(solution[0])
    alpha = solution[1]
    lr = solution[2]
    
    mlp = MLPRegressor(hidden_layer_sizes=(h_size,), alpha=alpha, 
                       learning_rate_init=lr, max_iter=200, random_state=42)
    mlp.fit(data['X_train'], data['y_train'])
    y_pred = mlp.predict(data['X_test'])
    return mean_squared_error(data['y_test'], y_pred)

def fitness_gmdh(solution, data):
    limit = solution[0]
    k_best = max(3, int(solution[1]))
    
    model = gmdh.Mia()
    model.fit(data['X_train'], data['y_train'], limit=limit, k_best=k_best)
    y_pred = model.predict(data['X_test'])
    return mean_squared_error(data['y_test'], y_pred)

def run_optimization(optimizer_class, fitness_func, data, bounds, name):
    problem = Problem(
        obj_func=lambda sol: fitness_func(sol, data),
        bounds=bounds,
        minmax="min",
    )
    
    model = optimizer_class(epoch=10, pop_size=5)
    best_agent = model.solve(problem)
    
    return best_agent.target.fitness, best_agent.solution, model.history.list_global_best_fit

if __name__ == "__main__":
    data = joblib.load('processed_data_improved/data.pkl')
    results = []
    convergence_curves = {}
    
    # MLP Optimization Bounds
    mlp_bounds = [
        IntegerVar(lb=10, ub=100, name="hidden_layer_size"),
        FloatVar(lb=0.0001, ub=0.01, name="alpha"),
        FloatVar(lb=0.001, ub=0.01, name="learning_rate_init")
    ]
    
    optimizers = [
        (MFO.OriginalMFO, "MFO"),
        (WOA.OriginalWOA, "ZHA"),
        (CSA.OriginalCSA, "CSA")
    ]
    
    for opt_class, opt_name in optimizers:
        print(f"Running {opt_name}-MLP...")
        fitness, solution, history = run_optimization(opt_class, fitness_mlp, data, mlp_bounds, opt_name)
        
        h_size = int(solution[0])
        alpha = solution[1]
        lr = solution[2]
        
        final_model = MLPRegressor(hidden_layer_sizes=(h_size,), alpha=alpha, 
                                   learning_rate_init=lr, max_iter=500, random_state=42)
        final_model.fit(data['X_train'], data['y_train'])
        y_pred = final_model.predict(data['X_test'])
        mae, rmse, r2 = evaluate_model(data['y_test'], y_pred)
        
        results.append({
            'model_name': f'{opt_name}-MLP',
            'best_params': {'hidden_layer_sizes': h_size, 'alpha': alpha, 'learning_rate_init': lr},
            'mae': mae, 'rmse': rmse, 'r2': r2, 'y_pred': y_pred
        })
        convergence_curves[f'{opt_name}-MLP'] = history

    # GMDH Optimization Bounds
    gmdh_bounds = [
        FloatVar(lb=0.01, ub=0.5, name="limit"),
        IntegerVar(lb=3, ub=10, name="k_best")
    ]
    
    for opt_class, opt_name in optimizers:
        print(f"Running {opt_name}-GMDH...")
        fitness, solution, history = run_optimization(opt_class, fitness_gmdh, data, gmdh_bounds, opt_name)
        
        limit = solution[0]
        k_best = max(3, int(solution[1]))
        
        final_model = gmdh.Mia()
        final_model.fit(data['X_train'], data['y_train'], limit=limit, k_best=k_best)
        y_pred = final_model.predict(data['X_test'])
        mae, rmse, r2 = evaluate_model(data['y_test'], y_pred)
        
        results.append({
            'model_name': f'{opt_name}-GMDH',
            'best_params': {'limit': limit, 'k_best': k_best},
            'mae': mae, 'rmse': rmse, 'r2': r2, 'y_pred': y_pred
        })
        convergence_curves[f'{opt_name}-GMDH'] = history

    os.makedirs('results_improved', exist_ok=True)
    joblib.dump(results, 'results_improved/optimized_results.pkl')
    joblib.dump(convergence_curves, 'results_improved/convergence_curves.pkl')
    print("Improved optimization complete.")
