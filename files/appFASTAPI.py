from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# ============================================================
# 1. Initialize FastAPI app
# ============================================================
app = FastAPI(
    title="Student Placement & Salary Predictor API",
    description="API untuk memprediksi status penempatan kerja mahasiswa dan estimasi gaji menggunakan LightGBM + XGBoost.",
    version="1.0.0"
)

# ============================================================
# 2. Load kedua model .pkl
# ============================================================
try:
    classifier_model = joblib.load("models/classifier_model.pkl")
    regressor_model  = joblib.load("models/regressor_model.pkl")
    print("✅ Kedua model berhasil dimuat!")
except Exception as e:
    classifier_model = None
    regressor_model  = None
    print(f"❌ Gagal memuat model: {e}")

# ============================================================
# 3. Input Schema (Pydantic)
# ============================================================
class StudentData(BaseModel):
    ssc_percentage:           float = Field(..., ge=0, le=100, description="Nilai ujian kelas 10 (%)")
    hsc_percentage:           float = Field(..., ge=0, le=100, description="Nilai ujian kelas 12 (%)")
    degree_percentage:        float = Field(..., ge=0, le=100, description="Nilai rata-rata perkuliahan (%)")
    cgpa:                     float = Field(..., ge=0, le=10,  description="IPK skala 10")
    entrance_exam_score:      float = Field(..., ge=0, le=100, description="Skor ujian masuk kerja")
    technical_skill_score:    float = Field(..., ge=0, le=100, description="Kemampuan teknis (0–100)")
    soft_skill_score:         float = Field(..., ge=0, le=100, description="Kemampuan soft skill (0–100)")
    internship_count:         int   = Field(..., ge=0,         description="Jumlah magang")
    live_projects:            int   = Field(..., ge=0,         description="Jumlah proyek nyata")
    work_experience_months:   int   = Field(..., ge=0,         description="Pengalaman kerja (bulan)")
    certifications:           int   = Field(..., ge=0,         description="Jumlah sertifikasi")
    attendance_percentage:    float = Field(..., ge=0, le=100, description="Persentase kehadiran")
    gender:                   str   = Field(...,               description="'Male' atau 'Female'")
    extracurricular_activities: str = Field(...,               description="'Yes' atau 'No'")

    class Config:
        json_schema_extra = {
            "example": {
                "ssc_percentage":             72.0,
                "hsc_percentage":             68.5,
                "degree_percentage":          75.0,
                "cgpa":                       7.8,
                "entrance_exam_score":        65.0,
                "technical_skill_score":      78.0,
                "soft_skill_score":           72.0,
                "internship_count":           2,
                "live_projects":              3,
                "work_experience_months":     6,
                "certifications":             3,
                "attendance_percentage":      82.0,
                "gender":                     "Male",
                "extracurricular_activities": "Yes"
            }
        }

# ============================================================
# 4. Endpoints
# ============================================================

@app.get("/")
def read_root():
    """Health check — verifikasi API berjalan."""
    return {
        "message": "Student Placement & Salary Predictor API aktif.",
        "docs":    "Kunjungi /docs untuk mencoba endpoint secara interaktif."
    }


@app.post("/predict")
def predict_placement(data: StudentData):
    """
    Prediksi status penempatan kerja dan estimasi gaji mahasiswa.

    - **placement_status**: 1 = Placed, 0 = Not Placed
    - **salary_lpa**: Estimasi gaji dalam LPA (hanya muncul jika placed)
    """
    if classifier_model is None or regressor_model is None:
        raise HTTPException(
            status_code=500,
            detail="Model belum termuat di server. Periksa file .pkl."
        )

    try:
        input_df = pd.DataFrame([data.dict()])

        # — Klasifikasi: placed atau tidak —
        placement = int(classifier_model.predict(input_df)[0])

        result = {
            "status":           "success",
            "placement_status": placement,
            "label":            "Placed" if placement == 1 else "Not Placed"
        }

        # — Regresi: hanya jika placed —
        if placement == 1:
            salary = float(regressor_model.predict(input_df)[0])
            result["salary_lpa"] = round(salary, 2)

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")