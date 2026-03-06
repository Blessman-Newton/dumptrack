import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
<<<<<<< HEAD
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import os

def preprocess_data_improved(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values
    if df.isnull().values.any():
        df = df.dropna()
    
    # 1. Outlier Treatment (Capping/Winsorization)
    # Using IQR to cap outliers
    features_to_cap = ['FullHaul', 'DumpingTime', 'EmptyHaul', 'LoadingTime', 'CycleDistance', 'TotalCycle']
    for col in features_to_cap:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # 2. Feature Engineering
    # - Loading Rate: MinedTonnes / LoadingTime
    # - Average Speed: CycleDistance / TotalCycle
    # - Efficiency: (LoadingTime + DumpingTime) / TotalCycle
    # Note: Avoid division by zero
    df['LoadingRate'] = df['MinedTonnes'] / (df['LoadingTime'] + 1e-6)
    df['AvgSpeed'] = df['CycleDistance'] / (df['TotalCycle'] + 1e-6)
    df['OperationalEfficiency'] = (df['LoadingTime'] + df['DumpingTime']) / (df['TotalCycle'] + 1e-6)
    
    # 3. Target Transformation
    # Given the right-skewed distribution, log transformation is recommended.
    # Use log1p to handle any potential zero values safely.
=======
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_improved(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # 1. Outlier Treatment (Capping at 1st and 99th percentiles)
    for col in df.columns:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = np.clip(df[col], lower, upper)
    
    # 2. Feature Engineering
    # - Haul Efficiency: CycleDistance / TotalCycle
    df['HaulEfficiency'] = df['CycleDistance'] / (df['TotalCycle'] + 1e-6)
    # - Loading Ratio: LoadingTime / TotalCycle
    df['LoadingRatio'] = df['LoadingTime'] / (df['TotalCycle'] + 1e-6)
    # - Dumping Ratio: DumpingTime / TotalCycle
    df['DumpingRatio'] = df['DumpingTime'] / (df['TotalCycle'] + 1e-6)
    # - Average Speed: CycleDistance / (FullHaul + EmptyHaul)
    df['AvgSpeed'] = df['CycleDistance'] / (df['FullHaul'] + df['EmptyHaul'] + 1e-6)
    
    # 3. Target Transformation (Log transformation)
    # Since MinedTonnes is positive, we can use log
>>>>>>> e3c5121 (Update with improved data pipeline and re-evaluated models)
    df['MinedTonnes_Log'] = np.log1p(df['MinedTonnes'])
    
    # Define features and target
    X = df.drop(['MinedTonnes', 'MinedTonnes_Log'], axis=1)
    y = df['MinedTonnes_Log']
<<<<<<< HEAD
    y_orig = df['MinedTonnes']
    
    # Split data
    X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = train_test_split(
        X, y, y_orig, test_size=0.2, random_state=42
    )
    
    # 4. Robust Scaling
    # Since we still have some variability, RobustScaler might be better than StandardScaler
    scaler_X = RobustScaler()
    scaler_y = StandardScaler() # Target is already log-transformed, so StandardScaler is fine
=======
    
    # Split data into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Perform feature scaling
    scaler_X = StandardScaler()
    # No need to scale y if we use log transformation, but we'll keep track of it
>>>>>>> e3c5121 (Update with improved data pipeline and re-evaluated models)
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
<<<<<<< HEAD
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    
=======
>>>>>>> e3c5121 (Update with improved data pipeline and re-evaluated models)
    # Save processed data and scalers
    data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
<<<<<<< HEAD
        'y_train': y_train_scaled,
        'y_test': y_test_scaled,
        'y_train_orig': y_train_orig.values,
        'y_test_orig': y_test_orig.values,
        'feature_names': X.columns.tolist(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'target_transform': 'log1p'
    }
    
    os.makedirs('processed_data', exist_ok=True)
    joblib.dump(data, 'processed_data/data_improved.pkl')
    print("Improved data preprocessing complete. Saved to processed_data/data_improved.pkl")
    return data

if __name__ == "__main__":
    preprocess_data_improved('cleaned_data.csv')
=======
        'y_train': y_train.values,
        'y_test': y_test.values,
        'feature_names': X.columns.tolist(),
        'scaler_X': scaler_X,
        'target_transform': 'log1p'
    }
    
    os.makedirs('processed_data_improved', exist_ok=True)
    joblib.dump(data, 'processed_data_improved/data.pkl')
    print("Improved data preprocessing complete. Saved to processed_data_improved/data.pkl")
    return data

if __name__ == "__main__":
    preprocess_improved('cleaned_data.csv')
>>>>>>> e3c5121 (Update with improved data pipeline and re-evaluated models)
