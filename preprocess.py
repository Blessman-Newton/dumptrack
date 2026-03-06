import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


BASE_FEATURES = [
    'FullHaul',
    'DumpingTime',
    'EmptyHaul',
    'LoadingTime',
    'CycleDistance',
    'TotalCycle',
]


def _cap_outliers_iqr(df, columns):
    """Cap outliers using IQR-based winsorization."""
    df_capped = df.copy()
    for col in columns:
        q1 = df_capped[col].quantile(0.25)
        q3 = df_capped[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)
    return df_capped


def _engineer_features(df):
    """Create additional operational features recommended by EDA."""
    eps = 1e-6
    df_feat = df.copy()

    # Engineering based on cycle and loading dynamics.
    df_feat['LoadingRate'] = df_feat['MinedTonnes'] / (df_feat['LoadingTime'] + eps)
    df_feat['AvgSpeed'] = df_feat['CycleDistance'] / (df_feat['TotalCycle'] + eps)
    df_feat['OperationalEfficiency'] = (
        (df_feat['LoadingTime'] + df_feat['DumpingTime']) / (df_feat['TotalCycle'] + eps)
    )

    return df_feat


def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    if df.isnull().values.any():
        df = df.dropna()

    # 1) Outlier treatment
    df = _cap_outliers_iqr(df, BASE_FEATURES)

    # 2) Feature engineering
    df = _engineer_features(df)

    # 3) Target transformation
    df['MinedTonnes_Log'] = np.log1p(df['MinedTonnes'])

    X = df.drop(['MinedTonnes', 'MinedTonnes_Log'], axis=1)
    y_log = df['MinedTonnes_Log']
    y_orig = df['MinedTonnes']

    X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = train_test_split(
        X, y_log, y_orig, test_size=0.2, random_state=42
    )

    scaler_X = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'y_train_orig': y_train_orig.values,
        'y_test_orig': y_test_orig.values,
        'feature_names': X.columns.tolist(),
        'scaler_X': scaler_X,
        'target_transform': 'log1p',
    }

    os.makedirs('processed_data', exist_ok=True)
    joblib.dump(data, 'processed_data/data.pkl')
    print('Improved preprocessing complete. Saved to processed_data/data.pkl')
    return data


if __name__ == '__main__':
    preprocess_data('cleaned_data.csv')
