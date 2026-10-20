import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Placement Predictor — Frontend",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .result-placed {
        background: linear-gradient(135deg, #d5f5e3, #a9dfbf);
        border-left: 5px solid #27ae60;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }
    .result-notplaced {
        background: linear-gradient(135deg, #fadbd8, #f5b7b1);
        border-left: 5px solid #e74c3c;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }
    .result-title   { font-size: 1.3rem; font-weight: 700; margin: 0; }
    .result-subtitle{ font-size: 0.88rem; margin-top: 4px; opacity: 0.75; }
    .api-box {
        background: rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 8px;
        padding: 10px 14px;
        font-family: monospace;
        font-size: 0.82rem;
        margin-top: 8px;
    }
    div.stButton > button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PRESET SCENARIOS
# ============================================================
SCENARIOS = {
    "🏆 Skenario 1 — Mahasiswa Berprestasi (Expected: Placed)": {
        "ssc_percentage":             88.0,
        "hsc_percentage":             85.0,
        "degree_percentage":          87.0,
        "cgpa":                       9.2,
        "entrance_exam_score":        90.0,
        "technical_skill_score":      92.0,
        "soft_skill_score":           88.0,
        "internship_count":           4,
        "live_projects":              5,
        "work_experience_months":     12,
        "certifications":             6,
        "attendance_percentage":      95.0,
        "gender":                     "Female",
        "extracurricular_activities": "Yes"
    },
    "📉 Skenario 2 — Mahasiswa Kesulitan (Expected: Not Placed)": {
        "ssc_percentage":             45.0,
        "hsc_percentage":             42.0,
        "degree_percentage":          48.0,
        "cgpa":                       4.1,
        "entrance_exam_score":        30.0,
        "technical_skill_score":      35.0,
        "soft_skill_score":           40.0,
        "internship_count":           0,
        "live_projects":              0,
        "work_experience_months":     0,
        "certifications":             0,
        "attendance_percentage":      52.0,
        "gender":                     "Male",
        "extracurricular_activities": "No"
    }
}

# ============================================================
# HELPER: panggil FastAPI
# ============================================================
def call_api(payload: dict):
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        return resp.status_code, resp.json()
    except requests.exceptions.ConnectionError:
        return None, {"error": "Tidak dapat terhubung ke FastAPI. Pastikan server berjalan di port 8000."}
    except Exception as e:
        return None, {"error": str(e)}

# ============================================================
# HELPER: tampilkan hasil prediksi
# ============================================================
def show_result(status_code, result, payload):
    if status_code is None:
        st.error(f"❌ Koneksi gagal: {result.get('error')}")
        return

    if status_code != 200:
        st.error(f"❌ API Error {status_code}: {result.get('detail')}")
        return

    placed  = result.get("placement_status")
    label   = result.get("label")
    salary  = result.get("salary_lpa")

    rcol1, rcol2 = st.columns([1, 1])

    with rcol1:
        if placed == 1:
            st.markdown(f"""
            <div class="result-placed">
                <p class="result-title">✅ {label}</p>
                <p class="result-subtitle">Mahasiswa ini berpeluang tinggi mendapat penempatan kerja.</p>
            </div>
            """, unsafe_allow_html=True)

            st.metric("💰 Estimasi Gaji", f"{salary:.2f} LPA")

            fig = go.Figure(go.Indicator(
                mode   = "gauge+number",
                value  = salary,
                title  = {"text": "Estimasi Gaji (LPA)", "font": {"size": 13}},
                number = {"suffix": " LPA", "font": {"size": 22}},
                gauge  = {
                    "axis": {"range": [0, 15]},
                    "bar":  {"color": "#27ae60", "thickness": 0.3},
                    "steps": [
                        {"range": [0,  5],  "color": "#fadbd8"},
                        {"range": [5,  10], "color": "#fdebd0"},
                        {"range": [10, 15], "color": "#d5f5e3"},
                    ]
                }
            ))
            fig.update_layout(
                height=240, margin=dict(t=50, b=10, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown(f"""
            <div class="result-notplaced">
                <p class="result-title">❌ {label}</p>
                <p class="result-subtitle">Tingkatkan skill dan pengalaman untuk meningkatkan peluang.</p>
            </div>
            """, unsafe_allow_html=True)
            st.info("Salary tidak diprediksi karena mahasiswa diprediksi tidak placed.")

    with rcol2:
        # Radar profil
        labels = ["SSC", "HSC", "Degree", "CGPA×10", "Technical", "Soft Skill", "Attendance"]
        vals   = [
            payload["ssc_percentage"],
            payload["hsc_percentage"],
            payload["degree_percentage"],
            payload["cgpa"] * 10,
            payload["technical_skill_score"],
            payload["soft_skill_score"],
            payload["attendance_percentage"]
        ]
        labels += [labels[0]]; vals += [vals[0]]

        fig_r = go.Figure(go.Scatterpolar(
            r=vals, theta=labels, fill="toself",
            name="Profil", line=dict(color="#3498db", width=2),
            fillcolor="rgba(52,152,219,0.2)"
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=[80]*8, theta=labels, fill="toself", name="Ideal",
            line=dict(color="#27ae60", width=1.5, dash="dot"),
            fillcolor="rgba(39,174,96,0.07)"
        ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Radar Profil Akademik & Skill",
            height=320, margin=dict(t=50, b=20, l=20, r=20),
            legend=dict(orientation="h", y=-0.1),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_r, use_container_width=True)

        # Raw API response
        with st.expander("🔌 Raw API Response"):
            st.markdown(f"""
            <div class="api-box">
            POST {API_URL}/predict<br>
            Status: {status_code}<br><br>
            {result}
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("# 🎓 Placement Predictor")
    st.caption("Frontend → FastAPI Backend")
    st.divider()

    # Status API
    st.markdown("### 🔌 Status API")
    try:
        ping = requests.get(f"{API_URL}/", timeout=3)
        if ping.status_code == 200:
            st.success("FastAPI **Online** ✅")
        else:
            st.warning("API merespons tapi tidak normal.")
    except:
        st.error("FastAPI **Offline** ❌\n\nJalankan:\n```\nuvicorn appFASTAPI:app --reload\n```")

    st.divider()
    st.markdown("### ℹ️ Info")
    st.markdown(f"**Endpoint:** `{API_URL}/predict`")
    st.markdown("**Method:** `POST`")
    st.markdown("**Format:** `application/json`")
    st.divider()
    st.markdown("### 💡 Tips")
    st.info("Gunakan tombol skenario di atas form untuk mengisi otomatis dan melihat dua hasil yang berbeda.")

# ============================================================
# MAIN
# ============================================================
st.title("🎓 Student Placement Predictor")
st.markdown("Frontend Streamlit → Backend FastAPI. Isi form manual atau gunakan skenario preset.")

# ── Tombol Skenario ─────────────────────────────────────────
st.subheader("⚡ Uji Coba Skenario Preset")
scol1, scol2 = st.columns(2)

scenario_data = None
active_scenario = None

with scol1:
    if st.button("🏆 Skenario 1 — Berprestasi", use_container_width=True):
        active_scenario = list(SCENARIOS.keys())[0]
        scenario_data   = SCENARIOS[active_scenario]
        st.session_state["preset"] = scenario_data

with scol2:
    if st.button("📉 Skenario 2 — Kesulitan", use_container_width=True):
        active_scenario = list(SCENARIOS.keys())[1]
        scenario_data   = SCENARIOS[active_scenario]
        st.session_state["preset"] = scenario_data

# Jalankan prediksi langsung jika skenario ditekan
if scenario_data:
    st.divider()
    st.subheader(f"📊 Hasil — {active_scenario}")
    code, result = call_api(scenario_data)
    show_result(code, result, scenario_data)

st.divider()

# ── Form Manual ─────────────────────────────────────────────
preset = st.session_state.get("preset", {})

st.subheader("📝 Input Manual")
with st.form("manual_form"):
    st.markdown("**📚 Akademik**")
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        ssc    = st.slider("SSC %",    0.0, 100.0, float(preset.get("ssc_percentage", 70.0)),    0.5)
        hsc    = st.slider("HSC %",    0.0, 100.0, float(preset.get("hsc_percentage", 70.0)),    0.5)
    with ac2:
        degree = st.slider("Degree %", 0.0, 100.0, float(preset.get("degree_percentage", 75.0)), 0.5)
        cgpa   = st.slider("CGPA",     0.0, 10.0,  float(preset.get("cgpa", 7.0)),               0.1)
    with ac3:
        entrance = st.slider("Entrance Exam", 0.0, 100.0, float(preset.get("entrance_exam_score", 60.0)), 0.5)

    st.markdown("**💻 Skills & Pengalaman**")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        technical = st.slider("Technical Skill", 0.0, 100.0, float(preset.get("technical_skill_score", 70.0)), 0.5)
        soft      = st.slider("Soft Skill",      0.0, 100.0, float(preset.get("soft_skill_score", 70.0)),      0.5)
    with sc2:
        internship = st.number_input("Internship",    0, 10, int(preset.get("internship_count", 1)))
        projects   = st.number_input("Live Projects", 0, 10, int(preset.get("live_projects", 1)))
    with sc3:
        work_exp       = st.number_input("Work Exp (bulan)", 0, 60, int(preset.get("work_experience_months", 0)))
        certifications = st.number_input("Certifications",   0, 20, int(preset.get("certifications", 2)))

    st.markdown("**📅 Kehadiran & Personal**")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        attendance = st.slider("Attendance %", 0.0, 100.0, float(preset.get("attendance_percentage", 75.0)), 0.5)
    with pc2:
        gender         = st.selectbox("Gender", ["Male", "Female"],
                                      index=0 if preset.get("gender", "Male") == "Male" else 1)
    with pc3:
        extracurricular = st.selectbox("Extracurricular", ["Yes", "No"],
                                       index=0 if preset.get("extracurricular_activities", "Yes") == "Yes" else 1)

    submitted = st.form_submit_button("🚀 Kirim ke API & Prediksi", type="primary", use_container_width=True)

if submitted:
    payload = {
        "ssc_percentage":             ssc,
        "hsc_percentage":             hsc,
        "degree_percentage":          degree,
        "cgpa":                       cgpa,
        "entrance_exam_score":        entrance,
        "technical_skill_score":      technical,
        "soft_skill_score":           soft,
        "internship_count":           internship,
        "live_projects":              projects,
        "work_experience_months":     work_exp,
        "certifications":             certifications,
        "attendance_percentage":      attendance,
        "gender":                     gender,
        "extracurricular_activities": extracurricular
    }

    st.divider()
    st.subheader("📊 Hasil — Input Manual")
    code, result = call_api(payload)
    show_result(code, result, payload)