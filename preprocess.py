import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values (if any)
    if df.isnull().values.any():
        df = df.dropna()
    
    # Define features and target
    # Based on the header: FullHaul,DumpingTime,EmptyHaul,LoadingTime,CycleDistance,TotalCycle,MinedTonnes
    # Target is likely MinedTonnes (Dump Truck Production)
    X = df.drop('MinedTonnes', axis=1)
    y = df['MinedTonnes']
    
    # Split data into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Perform feature scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Reshape y for scaler
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    
    # Save processed data and scalers
    data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_scaled,
        'y_test': y_test_scaled,
        'feature_names': X.columns.tolist(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    
    os.makedirs('processed_data', exist_ok=True)
    joblib.dump(data, 'processed_data/data.pkl')
    print("Data preprocessing complete. Saved to processed_data/data.pkl")
    return data

if __name__ == "__main__":
    preprocess_data('cleaned_data.csv')
