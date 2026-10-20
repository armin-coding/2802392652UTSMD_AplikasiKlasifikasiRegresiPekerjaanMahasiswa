import pandas as pd
from pathlib import Path
import os


BASE_DIR = Path(__file__).parent

# Mengarah langsung ke file B.csv di folder utama
INPUT_PATH = BASE_DIR / "B.csv" 
OUTPUT_DIR = BASE_DIR / "ingested"
OUTPUT_PATH = OUTPUT_DIR / "dataset_ingested.csv"

def ingest_data():
    print("Memulai proses Data Ingestion...")
    
    # Buat folder output jika belum ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    

    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ File tidak ditemukan di {INPUT_PATH}. Pastikan file B.csv sudah ada di folder utama.")
    
   
    if df.empty:
        raise ValueError("❌ Dataset kosong! Periksa kembali file B.csv.")
        

    required_columns = ['placement_status', 'salary_package_lpa', 'student_id']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Dataset tidak valid! Kehilangan kolom krusial: {missing_cols}")
        
    initial_rows = df.shape[0]
    df = df.drop_duplicates(subset=['student_id'], keep='first')
    
    if initial_rows != df.shape[0]:
        print(f"⚠️ Ditemukan {initial_rows - df.shape[0]} data duplikat (student_id sama). Duplikat telah dibersihkan.")
        
    
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✅ Data Ingestion Selesai!")
    print(f"   - Total Baris : {df.shape[0]}")
    print(f"   - Total Kolom : {df.shape[1]}")
    print(f"   - Tersimpan di: {OUTPUT_PATH}\n")
    
    return OUTPUT_PATH

if __name__ == "__main__":
    ingest_data()