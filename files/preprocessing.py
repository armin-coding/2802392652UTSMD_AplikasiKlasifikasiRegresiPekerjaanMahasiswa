import pandas as pd
import os
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


BASE_DIR = Path(__file__).parent
INPUT_PATH = BASE_DIR / "ingested" / "dataset_ingested.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CLF_DIR = ARTIFACTS_DIR / "classification"
REG_DIR = ARTIFACTS_DIR / "regression"

def preprocess():
    print("Memulai proses Preprocessing...")
    
    
    os.makedirs(CLF_DIR, exist_ok=True)
    os.makedirs(REG_DIR, exist_ok=True)
    

    df = pd.read_csv(INPUT_PATH)
    
    
    # student_id dibuang agar model tidak mempelajari nomor urut/ID mahasiswa, backlogs buang juga karena tidak butuh info tidak lulus karena biaya
    X = df.drop(columns=["placement_status", "salary_package_lpa", "student_id", "backlogs"])
    
    # Target
    y_clf = df["placement_status"]
    y_reg = df["salary_package_lpa"]
    
    #split klasifikasi menggunakan stratify agar rasio imbalanced kelas 0 dan 1 tetap terjaga)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    #aplit data regresi tanpa stratify karena target berupa angka kontinu
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    
   
    numcol = X_train_clf.select_dtypes(include=['int64', 'float64']).columns.tolist()
    catcol = X_train_clf.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # scikitlearn preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numcol),
        ("cat", categorical_transformer, catcol)
    ])
    
    # simpan artifacts data ke masing-masing folder
    # data klasifikasi
    X_train_clf.to_csv(CLF_DIR / "X_train.csv", index=False)
    X_test_clf.to_csv(CLF_DIR / "X_test.csv", index=False)
    y_train_clf.to_csv(CLF_DIR / "y_train.csv", index=False)
    y_test_clf.to_csv(CLF_DIR / "y_test.csv", index=False)
    
    # data regresi
    X_train_reg.to_csv(REG_DIR / "X_train.csv", index=False)
    X_test_reg.to_csv(REG_DIR / "X_test.csv", index=False)
    y_train_reg.to_csv(REG_DIR / "y_train.csv", index=False)
    y_test_reg.to_csv(REG_DIR / "y_test.csv", index=False)
    
    # simpan objek preprocessor (Arsitekturnya saja, fit dilakukan di tahap training)
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.pkl")
    
    print(f"✅ Preprocessing selesai.")
    print(f"   - Data Klasifikasi tersimpan di: {CLF_DIR}")
    print(f"   - Data Regresi tersimpan di    : {REG_DIR}")
    print(f"   - Objek Preprocessor tersimpan di: {ARTIFACTS_DIR / 'preprocessor.pkl'}")

if __name__ == "__main__":
    preprocess()