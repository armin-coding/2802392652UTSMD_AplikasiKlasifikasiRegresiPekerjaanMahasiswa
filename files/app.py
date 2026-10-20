import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Reduce top padding */
    .block-container { padding-top: 1.5rem; }

    /* Result cards */
    .result-placed {
        background: linear-gradient(135deg, #d5f5e3, #a9dfbf);
        border-left: 5px solid #27ae60;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .result-notplaced {
        background: linear-gradient(135deg, #fadbd8, #f5b7b1);
        border-left: 5px solid #e74c3c;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .result-title { font-size: 1.4rem; font-weight: 700; margin: 0; }
    .result-subtitle { font-size: 0.9rem; margin-top: 4px; opacity: 0.75; }

    /* Metric cards di sidebar */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.07);
        border-radius: 8px;
        padding: 8px 12px;
    }

    /* Submit button */
    div.stButton > button[kind="primary"] {
        width: 100%;
        height: 3rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# LOAD MODELS — path sudah diperbaiki (model → models)

@st.cache_resource
def load_models():
    try:
        clf = joblib.load("models/classifier_model.pkl")
        reg = joblib.load("models/regressor_model.pkl")
        return clf, reg
    except FileNotFoundError as e:
        st.error(f"❌ Model tidak ditemukan: {e}")
        st.info("Pastikan folder `models/` berisi `classifier_model.pkl` dan `regressor_model.pkl`.")
        st.stop()

classifier_model, regressor_model = load_models()

 
# SIDEBAR
 
with st.sidebar:
    st.markdown("# 🎓 Placement Predictor")
    st.caption("Powered by LightGBM + XGBoost")
    st.divider()

    st.markdown("### 📋 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini memprediksi **peluang penempatan kerja** mahasiswa
    dan **estimasi gaji** menggunakan model machine learning
    yang dilatih dari data historis kampus.
    """)
    st.divider()

    st.markdown("### 📈 Performa Model")
    c1, c2 = st.columns(2)
    c1.metric("F1-Score", "≥ 0.60", help="Untuk model klasifikasi placement")
    c2.metric("R² Score", "≥ 0.33", help="Untuk model regresi gaji")
    st.divider()

    st.markdown("### 💡 Tips Pengisian")
    st.info("""
    - Isi semua kolom dengan data yang akurat
    - CGPA dalam skala **0–10**
    - Semua persentase dalam skala **0–100**
    - Skor teknis & soft skill dalam skala **0–100**
    """)
    st.divider()

    with st.expander("🔧 Debug: Daftar Fitur Model"):
        st.code("""
ssc_percentage
hsc_percentage
degree_percentage
cgpa
entrance_exam_score
technical_skill_score
soft_skill_score
internship_count
live_projects
work_experience_months
certifications
attendance_percentage
gender
extracurricular_activities
        """)

 
# MAIN CONTENT
 
st.title("🎓 Student Placement & Salary Predictor")
st.markdown("Gunakan form di bawah untuk mendapatkan prediksi penempatan kerja dan estimasi gaji secara instan.")

tab_predict, tab_insight = st.tabs(["🔍 Prediksi", "📊 Panduan & Insight"])

# ─── TAB 1: PREDIKSI ────────────────────────────────────────
with tab_predict:
    with st.form("prediction_form"):

        # — Akademik —
        st.subheader("📚 Performa Akademik")
        acol1, acol2, acol3 = st.columns(3)
        with acol1:
            ssc    = st.slider("SSC Percentage", 0.0, 100.0, 70.0, 0.5,
                               help="Nilai ujian kelas 10 (%)")
            hsc    = st.slider("HSC Percentage", 0.0, 100.0, 70.0, 0.5,
                               help="Nilai ujian kelas 12 (%)")
        with acol2:
            degree = st.slider("Degree Percentage", 0.0, 100.0, 75.0, 0.5,
                               help="Nilai rata-rata perkuliahan (%)")
            cgpa   = st.slider("CGPA", 0.0, 10.0, 7.0, 0.1,
                               help="IPK dalam skala 10")
        with acol3:
            entrance = st.slider("Entrance Exam Score", 0.0, 100.0, 60.0, 0.5,
                                 help="Skor ujian masuk kerja (0–100)")

        st.divider()

        # — Skill & Pengalaman —
        st.subheader("💻 Skills & Pengalaman")
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            technical = st.slider("Technical Skill Score", 0.0, 100.0, 70.0, 0.5)
            soft      = st.slider("Soft Skill Score",      0.0, 100.0, 70.0, 0.5)
        with scol2:
            internship     = st.number_input("Internship Count",             0, 10,  1)
            projects       = st.number_input("Live Projects",                0, 10,  1)
        with scol3:
            work_exp       = st.number_input("Work Experience (Months)",     0, 60,  0)
            certifications = st.number_input("Certifications",               0, 20,  2)

        st.divider()

        # — Kehadiran & Personal —
        st.subheader("📅 Kehadiran & Data Personal")
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            attendance     = st.slider("Attendance Percentage", 0.0, 100.0, 75.0, 0.5)
        with pcol2:
            gender         = st.selectbox("Gender", ["Male", "Female"])
        with pcol3:
            extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

        st.markdown("")
        submitted = st.form_submit_button("🔍 Prediksi Sekarang", type="primary")

    # ── Hasil Prediksi ──────────────────────────────────────
    if submitted:
        input_df = pd.DataFrame([{
            "ssc_percentage":          ssc,
            "hsc_percentage":          hsc,
            "degree_percentage":       degree,
            "cgpa":                    cgpa,
            "entrance_exam_score":     entrance,
            "technical_skill_score":   technical,
            "soft_skill_score":        soft,
            "internship_count":        internship,
            "live_projects":           projects,
            "work_experience_months":  work_exp,
            "certifications":          certifications,
            "attendance_percentage":   attendance,
            "gender":                  gender,
            "extracurricular_activities": extracurricular
        }])

        try:
            placement = classifier_model.predict(input_df)[0]

            st.divider()
            st.subheader("📊 Hasil Prediksi")

            rcol1, rcol2 = st.columns([1, 1])

            # — Kiri: Status & Gaji —
            with rcol1:
                if placement == 1:
                    st.markdown("""
                    <div class="result-placed">
                        <p class="result-title">✅ Likely to be PLACED</p>
                        <p class="result-subtitle">Profil mahasiswa ini berpeluang tinggi mendapat penempatan kerja.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    salary = regressor_model.predict(input_df)[0]

                    st.metric("💰 Estimasi Gaji", f"{salary:.2f} LPA",
                              help="Dalam satuan Lakh Per Annum (LPA)")

                    # Gauge chart gaji
                    fig_gauge = go.Figure(go.Indicator(
                        mode   = "gauge+number+delta",
                        value  = salary,
                        delta  = {"reference": 5.0, "suffix": " LPA"},
                        title  = {"text": "Estimasi Gaji (LPA)", "font": {"size": 14}},
                        number = {"suffix": " LPA", "font": {"size": 24}},
                        gauge  = {
                            "axis": {"range": [0, 15], "tickwidth": 1},
                            "bar":  {"color": "#27ae60", "thickness": 0.3},
                            "steps": [
                                {"range": [0,  5],  "color": "#fadbd8"},
                                {"range": [5,  10], "color": "#fdebd0"},
                                {"range": [10, 15], "color": "#d5f5e3"},
                            ],
                            "threshold": {
                                "line": {"color": "#e74c3c", "width": 3},
                                "thickness": 0.75,
                                "value": 5.0
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        height=260,
                        margin=dict(t=50, b=10, l=20, r=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#333"
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                else:
                    st.markdown("""
                    <div class="result-notplaced">
                        <p class="result-title">❌ Likely NOT Placed</p>
                        <p class="result-subtitle">Tingkatkan skill dan pengalaman untuk meningkatkan peluang.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.warning("💡 **Saran:** Fokus pada peningkatan CGPA, technical skill, dan tambah pengalaman magang.")

                    # Bar chart saran area perbaikan
                    gaps = {
                        "Metrik":  ["CGPA",   "Technical", "Soft Skill", "Internship", "Attendance"],
                        "Kamu":    [cgpa*10,  technical,   soft,         internship*10, attendance],
                        "Target":  [85,       80,          80,           30,            85]
                    }
                    df_gap = pd.DataFrame(gaps)
                    fig_gap = go.Figure()
                    fig_gap.add_bar(x=df_gap["Metrik"], y=df_gap["Kamu"],   name="Nilai Kamu",  marker_color="#3498db")
                    fig_gap.add_bar(x=df_gap["Metrik"], y=df_gap["Target"], name="Target",      marker_color="#e0e0e0")
                    fig_gap.update_layout(
                        barmode="group", title="Gap ke Target",
                        height=220, margin=dict(t=40, b=10, l=10, r=10),
                        legend=dict(orientation="h", y=-0.2),
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_gap, use_container_width=True)

            # — Kanan: Radar profil mahasiswa —
            with rcol2:
                radar_labels  = ["SSC", "HSC", "Degree", "CGPA×10", "Technical", "Soft Skill", "Attendance"]
                radar_values  = [ssc, hsc, degree, cgpa * 10, technical, soft, attendance]
                # Tutup polygon
                radar_labels += [radar_labels[0]]
                radar_values += [radar_values[0]]

                fig_radar = go.Figure(go.Scatterpolar(
                    r     = radar_values,
                    theta = radar_labels,
                    fill  = "toself",
                    name  = "Profil Kamu",
                    line  = dict(color="#3498db", width=2),
                    fillcolor="rgba(52,152,219,0.2)"
                ))
                # Tambah overlay profil ideal
                ideal = [80, 80, 80, 80, 80, 80, 80, 80]
                fig_radar.add_trace(go.Scatterpolar(
                    r     = ideal,
                    theta = radar_labels,
                    fill  = "toself",
                    name  = "Profil Ideal",
                    line  = dict(color="#27ae60", width=1.5, dash="dot"),
                    fillcolor="rgba(39,174,96,0.07)"
                ))
                fig_radar.update_layout(
                    polar    = dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title    = "Radar Profil Akademik & Skill",
                    height   = 360,
                    margin   = dict(t=50, b=20, l=20, r=20),
                    legend   = dict(orientation="h", y=-0.08),
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                # Tabel ringkasan input
                with st.expander("📋 Ringkasan Data Input"):
                    summary = pd.DataFrame({
                        "Fitur": [
                            "SSC %", "HSC %", "Degree %", "CGPA", "Entrance Score",
                            "Technical Skill", "Soft Skill", "Internship",
                            "Live Projects", "Work Exp (Mo)", "Certifications",
                            "Attendance %", "Gender", "Extracurricular"
                        ],
                        "Nilai": [
                            ssc, hsc, degree, cgpa, entrance,
                            technical, soft, internship,
                            projects, work_exp, certifications,
                            attendance, gender, extracurricular
                        ]
                    })
                    st.dataframe(summary, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error saat prediksi: {e}")
            st.exception(e)


# ─── TAB 2: PANDUAN & INSIGHT ───────────────────────────────
with tab_insight:
    st.subheader("📖 Panduan Fitur")

    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.markdown("""
        **Akademik**
        | Fitur | Keterangan |
        |---|---|
        | SSC Percentage | Nilai ujian kelas 10 (%) |
        | HSC Percentage | Nilai ujian kelas 12 (%) |
        | Degree Percentage | Nilai rata-rata kuliah (%) |
        | CGPA | IPK, skala 0–10 |
        | Entrance Exam Score | Skor ujian masuk kerja (0–100) |
        """)

    with gcol2:
        st.markdown("""
        **Skills & Pengalaman**
        | Fitur | Keterangan |
        |---|---|
        | Technical Skill Score | Kemampuan teknis, 0–100 |
        | Soft Skill Score | Komunikasi & kepemimpinan, 0–100 |
        | Internship Count | Jumlah program magang |
        | Live Projects | Proyek nyata yang diselesaikan |
        | Certifications | Jumlah sertifikasi resmi |
        """)

    st.divider()
    st.subheader("📊 Estimasi Kepentingan Fitur")

    factors_df = pd.DataFrame({
        "Fitur": [
            "CGPA", "Technical Skill", "Soft Skill", "Attendance",
            "Entrance Exam", "Internship", "Live Projects",
            "Degree %", "Work Exp", "Certifications"
        ],
        "Estimasi Importance (%)": [22, 18, 15, 12, 10, 8, 6, 4, 3, 2],
        "Kategori": [
            "Akademik", "Skill", "Skill", "Akademik",
            "Akademik", "Pengalaman", "Pengalaman",
            "Akademik", "Pengalaman", "Pengalaman"
        ]
    }).sort_values("Estimasi Importance (%)", ascending=True)

    color_map = {"Akademik": "#3498db", "Skill": "#27ae60", "Pengalaman": "#e67e22"}

    fig_bar = px.bar(
        factors_df, x="Estimasi Importance (%)", y="Fitur",
        orientation="h",
        color="Kategori",
        color_discrete_map=color_map,
        title="Estimasi Bobot Fitur terhadap Keputusan Placement",
        text="Estimasi Importance (%)"
    )
    fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_bar.update_layout(
        height=400,
        margin=dict(t=50, b=20, l=10, r=60),
        legend_title="Kategori",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("⚠️ *Estimasi berdasarkan domain knowledge. Feature importance aktual tersimpan di MLflow.*")

    st.divider()
    st.subheader("🏗️ Arsitektur Model")
    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        st.info("""
        **Klasifikasi — LightGBM**
        - Target: Placed (1) / Not Placed (0)
        - Handling imbalance: `scale_pos_weight`
        - Metric utama: F1-Score
        """)
    with arch_col2:
        st.success("""
        **Regresi — XGBoost**
        - Target: Salary Package (LPA)
        - Hanya dijalankan jika placed = 1
        - Metric utama: R² Score
        """)