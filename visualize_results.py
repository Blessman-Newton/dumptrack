import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor

def generate_plots():
    os.makedirs('plots', exist_ok=True)
    
    # Load results
    baseline_results = joblib.load('results/baseline_results.pkl')
    optimized_results = joblib.load('results/optimized_results.pkl')
    convergence_curves = joblib.load('results/convergence_curves.pkl')
    data = joblib.load('processed_data/data.pkl')
    
    all_results = baseline_results + optimized_results
    df_results = pd.DataFrame(all_results)
    
    # 1. Performance Comparison Table (Save as CSV)
    df_results[['model_name', 'mae', 'rmse', 'r2']].to_csv('results/performance_comparison.csv', index=False)
    
    # 2. Performance Bar Plots
    metrics = ['mae', 'rmse', 'r2']
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='model_name', y=metric, data=df_results)
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'plots/{metric}_comparison.png')
        plt.close()
    
    # 3. Convergence Curves
    plt.figure(figsize=(12, 6))
    for model_name, curve in convergence_curves.items():
        plt.plot(curve, label=model_name)
    plt.title('Convergence Curves of Optimization Algorithms')
    plt.xlabel('Epoch')
    plt.ylabel('Fitness (MSE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/convergence_curves.png')
    plt.close()
    
    # 4. Feature Importance (using the best MLP model as an example)
    # Find the best MLP model
    mlp_results = [r for r in all_results if 'MLP' in r['model_name']]
    best_mlp_info = min(mlp_results, key=lambda x: x['mae'])
    
    # Re-train the best MLP to get the model object
    params = best_mlp_info['best_params']
    if isinstance(params, dict) and 'hidden_layer_sizes' in params:
        h_size = params['hidden_layer_sizes']
        if isinstance(h_size, tuple):
            h_size_tuple = h_size
        else:
            h_size_tuple = (int(h_size),)
            
        best_mlp = MLPRegressor(hidden_layer_sizes=h_size_tuple, 
                                alpha=params.get('alpha', 0.0001), 
                                learning_rate_init=params.get('learning_rate_init', 0.001), 
                                max_iter=500, random_state=42)
    else:
        # Fallback for baseline if it's not a dict
        best_mlp = MLPRegressor(random_state=42)
        
    best_mlp.fit(data['X_train'], data['y_train'])
    
    result = permutation_importance(best_mlp, data['X_test'], data['y_test'], n_repeats=10, random_state=42)
    sorted_idx = result.importances_mean.argsort()
    
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(data['feature_names'])[sorted_idx], result.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance (Best MLP Model)")
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()
    
    print("Plots generated in plots/ directory.")

if __name__ == "__main__":
    generate_plots()
