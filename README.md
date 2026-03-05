# Dump Truck Production Prediction (dumptrack)

This project aims to predict dump truck production (`MinedTonnes`) using various operational features such as haul times, cycle distances, and loading times. The models have been significantly improved through feature engineering, outlier treatment, and target transformation.

## 1. Project Overview
The dataset contains operational data for dump trucks, including:
- `FullHaul`, `DumpingTime`, `EmptyHaul`, `LoadingTime`, `CycleDistance`, `TotalCycle`
- Target: `MinedTonnes` (Dump Truck Production)

## 2. Recent Improvements
Based on the EDA findings, the following enhancements were implemented:
- **Feature Engineering**: Added `LoadingRate`, `AvgSpeed`, and `OperationalEfficiency`.
- **Outlier Treatment**: Implemented IQR-based capping (Winsorization) and switched to `RobustScaler`.
- **Target Transformation**: Applied `log1p` transformation to the target variable to handle right-skewness.

## 3. Installation
Ensure you have the following dependencies installed:
```bash
pip install pandas numpy scikit-learn joblib gmdh
```

## 4. How to Run

### Step 1: Data Preprocessing
Run the improved preprocessing script to generate the engineered features and apply transformations:
```bash
python3 preprocess_improved.py
```
This will create `processed_data/data_improved.pkl`.

### Step 2: Model Training and Evaluation
Run the improved model training script to train the MLP model and compare its performance:
```bash
python3 baseline_models_improved.py
```

### (Optional) Running Baseline Models
To compare against the original baseline (without improvements):
```bash
python3 run_baseline.py
```

## 5. Performance Comparison
The improvements have led to a dramatic increase in model accuracy:

| Metric | Original Baseline | Improved Model |
| :--- | :--- | :--- |
| **MAE** | ~200+ | **27.23** |
| **RMSE** | ~300+ | **46.29** |
| **R²** | ~0.28 | **0.995** |

## 6. Files Description
- `preprocess_improved.py`: Implements new feature engineering and data treatment.
- `baseline_models_improved.py`: Trains the improved MLP model.
- `IMPROVEMENTS_SUMMARY.md`: Detailed report of the changes and results.
- `cleaned_data.csv`: The input dataset.

## 7. License
This project is for educational and research purposes.
