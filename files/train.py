import pandas as pd
import joblib
import os
from pathlib import Path
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

# Konfigurasi Path
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"

def train_classification(preprocessor):
    print("\n--- Memulai Training Model Klasifikasi ---")
    try:
        X_train = pd.read_csv(ARTIFACTS_DIR / "classification" / "X_train.csv")
        y_train = pd.read_csv(ARTIFACTS_DIR / "classification" / "y_train.csv")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Pastikan preprocessing.py sudah berjalan.")
        return

    #Hitung ratio secara dinamis seperti di ipynb
    counts = y_train.value_counts()
    ratio = float(counts.iloc[0] / counts.iloc[1])
    
    # panggil model pemenang Clasifikasi (LGBM)
    # scale_pos_weight=2 digunakan untuk menangani imbalanced data tanpa overfit
    clf_model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        scale_pos_weight=ratio,
        random_state=42,
        verbosity=-1 # Menghilangkan warning log yang tidak perlu
    )
    
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", clf_model)
    ])
    
    # MLflow Tracking untuk Klasifikasi
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Placement_Classification")
    with mlflow.start_run(run_name="XGBoost_Classifier_Final"):
        print("Sedang melatih algoritma klasifikasi...")
        full_pipeline.fit(X_train, y_train.values.ravel())
        
        # Log parameter ke mlflow
        mlflow.log_params(clf_model.get_params())
        
        # Simpan model (.pkl) ke MLflow
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(full_pipeline, MODELS_DIR / "classifier_model.pkl")
        mlflow.sklearn.log_model(full_pipeline, "classifier_pipeline")
        
        print("✅ Training Klasifikasi selesai! Model tersimpan di 'models/classifier_model.pkl'.")

def train_regression(preprocessor):
    print("\n--- Memulai Training Model Regresi ---")
    try:
        X_train = pd.read_csv(ARTIFACTS_DIR / "regression" / "X_train.csv")
        y_train = pd.read_csv(ARTIFACTS_DIR / "regression" / "y_train.csv")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Pastikan preprocessing.py sudah berjalan.")
        return

    # panggil Regresi (XGBoost hasil tuning)
    reg_model = XGBRegressor(
        learning_rate=0.01, 
        max_depth=3, 
        n_estimators=200, 
        random_state=42
    )
    
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", reg_model)
    ])
    
    # MLflow Tracking untuk Regresi
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Salary_Regression")
    with mlflow.start_run(run_name="XGBoost_Regressor_Final"):
        print("Sedang melatih algoritma regresi...")
        full_pipeline.fit(X_train, y_train.values.ravel())
        
        # log parameter ke MLflow
        mlflow.log_params(reg_model.get_params())
        
        # ssimpan model lokal (.pkl) dan ke MLflow
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(full_pipeline, MODELS_DIR / "regressor_model.pkl")
        mlflow.sklearn.log_model(full_pipeline, "regressor_pipeline")
        
        print("✅ Training Regresi selesai! Model tersimpan di 'models/regressor_model.pkl'.")

def run_all_training():
    print("Memuat objek preprocessor universal...")
    try:
        preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.pkl")
    except FileNotFoundError:
        print("❌ Error: preprocessor.pkl tidak ditemukan. Jalankan preprocessing.py terlebih dahulu.")
        return

    # TRAIN KEDUANYA
    train_classification(preprocessor)
    train_regression(preprocessor)
    print("\n🎉 Seluruh proses training berhasil diselesaikan!")

if __name__ == "__main__":
    run_all_training()