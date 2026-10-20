from dataingestion import ingest_data
from preprocessing import preprocess
from train import run_all_training
from evaluation import evaluate_classification, evaluate_regression

# THRESHOLD kita
F1_THRESHOLD = 0.60
R2_THRESHOLD = 0.33

def run_pipeline():
    print("🚀 MEMULAI EKSEKUSI END-TO-END ML PIPELINE 🚀\n")
    try:
        # 1. Ingestion
        ingest_data()
        print("-" * 50)
        
        # 2. Preprocessing
        preprocess()
        print("-" * 50)
        
        # 3. Training
        run_all_training()
        print("-" * 50)
        
        # 4. Evaluasi 
        f1_score_val = evaluate_classification()
        r2_score_val = evaluate_regression()
        print("-" * 50)
        
        # 5. Validasi
        print("📊 HASIL KEPUTUSAN DEPLOYMENT:")
        
        # klasifikasi
        if f1_score_val >= F1_THRESHOLD:
            print(f"  [KLASIFIKASI] ✅ APPROVED. (F1-Score: {f1_score_val:.4f} >= {F1_THRESHOLD})")
        else:
            print(f"  [KLASIFIKASI] ⚠️ REJECTED. (F1-Score: {f1_score_val:.4f} < {F1_THRESHOLD})")
            
        # regresi
        if r2_score_val >= R2_THRESHOLD:
            print(f"  [REGRESI]     ✅ APPROVED. (R2-Score: {r2_score_val:.4f} >= {R2_THRESHOLD})")
        else:
            print(f"  [REGRESI]     ⚠️ REJECTED. (R2-Score: {r2_score_val:.4f} < {R2_THRESHOLD})")

        print("\n🏁 Eksekusi Pipeline Selesai 🏁")

    except Exception as e:
        print(f"\n❌ Pipeline gagal dieksekusi: {e}")

if __name__ == "__main__":
    run_pipeline()