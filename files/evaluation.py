import pandas as pd
import numpy as np
import joblib
import mlflow
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)



BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"

def evaluate_classification():
    print("\n--- Evaluasi Model Klasifikasi ---")
    try:
        X_test = pd.read_csv(ARTIFACTS_DIR / "classification" / "X_test.csv")
        y_test = pd.read_csv(ARTIFACTS_DIR / "classification" / "y_test.csv")
        model = joblib.load(MODELS_DIR / "classifier_model.pkl")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Pastikan train.py sudah berjalan.")
        return 0.0 # Return default jika gagal

    # Prediksi
    y_pred = model.predict(X_test)
    
    # Hitung Metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # MLflow 
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Placement_Classification")
    with mlflow.start_run(run_name="Evaluation_Classifier"):
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
        print("✅ Metrik Klasifikasi berhasil dicatat ke MLflow.")
        
    
    return f1








def evaluate_regression():
    print("\n--- Evaluasi Model Regresi ---")
    try:
        X_test = pd.read_csv(ARTIFACTS_DIR / "regression" / "X_test.csv")
        y_test = pd.read_csv(ARTIFACTS_DIR / "regression" / "y_test.csv")
        model = joblib.load(MODELS_DIR / "regressor_model.pkl")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Pastikan train.py sudah berjalan.")
        return 0.0 

    # Prediksi
    y_pred = model.predict(X_test)
    
    # Hitung Metrik
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"R2 Score : {r2:.4f}")
    
    # MLflow Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Salary_Regression")
    with mlflow.start_run(run_name="Evaluation_Regressor"):
        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2
        })
        print("✅ Metrik Regresi berhasil dicatat ke MLflow.")
        
    return r2




if __name__ == "__main__":
    evaluate_classification()
    evaluate_regression()