"""
app.py
─────────────────────────────────────────────────────────────────────────────
MODULE 5 — STREAMLIT WEB INTERFACE
Medicine Recommendation System — Complete UI
─────────────────────────────────────────────────────────────────────────────
"""

import os, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediRec AI — Medicine Recommendation System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px; font-weight: 500; }

/* Main background */
.main { background: #0f172a; color: #e2e8f0; }

/* Cards */
.med-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.med-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(59,130,246,0.2);
}
.med-card .rank-badge {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    color: white;
    border-radius: 50%;
    width: 32px; height: 32px;
    display: inline-flex;
    align-items: center; justify-content: center;
    font-weight: 700; font-size: 14px;
    margin-right: 10px;
}
.med-name { font-size: 18px; font-weight: 700; color: #93c5fd; }
.med-desc { color: #94a3b8; font-size: 14px; margin: 8px 0; }
.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    margin: 3px;
}
.pill-green  { background: #064e3b; color: #6ee7b7; }
.pill-yellow { background: #451a03; color: #fcd34d; }
.pill-red    { background: #450a0a; color: #fca5a5; }
.pill-blue   { background: #1e3a5f; color: #93c5fd; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #312e81 50%, #1e293b 100%);
    border-radius: 20px;
    padding: 48px 40px;
    margin-bottom: 30px;
    text-align: center;
    border: 1px solid #334155;
}
.hero h1 { font-size: 42px; font-weight: 700; color: #e2e8f0; margin: 0; }
.hero p  { font-size: 18px; color: #94a3b8; margin-top: 12px; }

/* Metric cards */
.metric-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}
.metric-box .value { font-size: 28px; font-weight: 700; color: #60a5fa; }
.metric-box .label { font-size: 13px; color: #94a3b8; margin-top: 4px; }

/* Step cards */
.step-card {
    background: #1e293b;
    border-left: 4px solid #3b82f6;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.step-card .step-num { color: #60a5fa; font-weight: 700; font-size: 13px; }
.step-card .step-text { color: #cbd5e1; font-size: 15px; margin-top: 4px; }

/* Disease badge */
.disease-badge {
    background: linear-gradient(90deg, #1d4ed8, #7c3aed);
    color: white;
    font-size: 22px;
    font-weight: 700;
    padding: 12px 28px;
    border-radius: 12px;
    display: inline-block;
    margin: 10px 0;
}
.confidence-bar-bg {
    background: #1e293b;
    border-radius: 999px;
    height: 12px;
    width: 100%;
    margin-top: 8px;
}
.confidence-bar-fill {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    border-radius: 999px;
    height: 12px;
}
/* Safety note */
.safety-note {
    background: #1c1917;
    border: 1px solid #78350f;
    border-radius: 10px;
    padding: 12px 16px;
    color: #fcd34d;
    font-size: 14px;
    margin: 6px 0;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading AI engine…")
def load_engine():
    try:
        from recommendation import MedicineRecommendationEngine
        return MedicineRecommendationEngine(), None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def load_metrics_df():
    path = os.path.join(MODEL_DIR, "all_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    rows = []
    for m in data:
        rows.append({
            "Model":     m["model"],
            "Accuracy":  m["accuracy"],
            "Precision": m["precision"],
            "Recall":    m["recall"],
            "F1-Score":  m["f1_score"],
            "ROC-AUC":   m["roc_auc"] if m["roc_auc"] != "N/A" else "—",
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:40px;'>💊</div>
        <div style='font-size:20px; font-weight:700; color:#60a5fa;'>MediRec AI</div>
        <div style='font-size:12px; color:#64748b; margin-top:4px;'>Medicine Recommendation System</div>
    </div>
    <hr style='border-color:#334155; margin: 10px 0 20px;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home", "🩺  Symptom Input", "📋  Results", "📊  Model Performance", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style='border-color:#334155; margin: 20px 0;'>
    <div style='font-size:12px; color:#475569; text-align:center;'>
        Powered by Scikit-learn & Keras<br>
        Final Year Project — 2025
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div class='hero'>
        <div style='font-size:52px;'>💊</div>
        <h1>MediRec AI</h1>
        <p>An intelligent Medicine Recommendation System powered by<br>
        Machine Learning & Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='metric-box'><div class='value'>20</div><div class='label'>Diseases Covered</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-box'><div class='value'>7</div><div class='label'>ML/DL Models</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-box'><div class='value'>60+</div><div class='label'>Medicines Mapped</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-box'><div class='value'>3000</div><div class='label'>Training Samples</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚙️ How It Works")

    steps = [
        ("STEP 1", "Enter your symptoms, age, gender and medical history"),
        ("STEP 2", "Our AI model predicts your most probable disease"),
        ("STEP 3", "The recommendation engine maps disease → medicines"),
        ("STEP 4", "Medicines are ranked by effectiveness & your safety profile"),
        ("STEP 5", "Review personalised recommendations with dosage & warnings"),
    ]
    for num, text in steps:
        st.markdown(f"""
        <div class='step-card'>
            <div class='step-num'>{num}</div>
            <div class='step-text'>{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("⚠️ **Medical Disclaimer**: This system is for educational purposes only. Always consult a qualified healthcare professional before taking any medication.")

    col_start, _ = st.columns([1, 3])
    with col_start:
        if st.button("🩺 Get Started →", use_container_width=True, type="primary"):
            st.session_state["page"] = "🩺  Symptom Input"
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SYMPTOM INPUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🩺  Symptom Input":
    st.markdown("## 🩺 Patient Information & Symptom Selection")
    st.caption("Fill in your details and select all applicable symptoms.")

    engine, err = load_engine()
    if err:
        st.error(f"**Engine not ready.** {err}")
        st.info("Run `python train.py` from the project folder first, then refresh.")
        st.stop()

    all_symptoms = engine.get_all_symptoms()

    with st.form("patient_form"):
        # ── Demographics ──────────────────────────────────────────────────
        st.markdown("#### 👤 Demographics")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        st.markdown("#### 🔴 Symptoms *(select all that apply)*")
        # Group symptoms alphabetically into 3 columns
        cols = st.columns(3)
        selected_symptoms = []
        for i, sym in enumerate(all_symptoms):
            label = sym.replace("_", " ").title()
            if cols[i % 3].checkbox(label, key=f"sym_{sym}"):
                selected_symptoms.append(sym)

        st.markdown("#### 📋 Medical History")
        c3, c4 = st.columns(2)
        with c3:
            history_input = st.text_area(
                "Past medical history (comma-separated)",
                placeholder="e.g. hypertension, diabetes",
                height=80,
            )
        with c4:
            conditions_input = st.text_area(
                "Existing conditions (comma-separated)",
                placeholder="e.g. kidney disease, pregnancy",
                height=80,
            )

        submitted = st.form_submit_button("💊 Get Recommendation", use_container_width=True, type="primary")

    if submitted:
        if not selected_symptoms:
            st.error("⚠️ Please select at least one symptom.")
        else:
            history    = [h.strip() for h in history_input.split(",") if h.strip()]
            conditions = [c.strip() for c in conditions_input.split(",") if c.strip()]

            from recommendation import PatientProfile
            profile = PatientProfile(
                symptoms             = selected_symptoms,
                age                  = age,
                gender               = gender,
                medical_history      = history,
                existing_conditions  = conditions,
            )
            st.session_state["profile"] = profile
            st.session_state["result"]  = engine.recommend(profile)
            st.success("✅ Recommendation ready! Navigate to **📋 Results**.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RESULTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📋  Results":
    st.markdown("## 📋 Diagnosis & Medicine Recommendations")

    if "result" not in st.session_state:
        st.warning("No recommendation yet. Go to **🩺 Symptom Input** first.")
        st.stop()

    result  = st.session_state["result"]
    profile = st.session_state["profile"]

    # ── Disease prediction ────────────────────────────────────────────────
    st.markdown("### 🎯 Predicted Diagnosis")
    conf_pct = int(result.confidence_score * 100)

    col_d, col_c = st.columns([2, 1])
    with col_d:
        st.markdown(f"<div class='disease-badge'>🦠 {result.predicted_disease}</div>", unsafe_allow_html=True)
    with col_c:
        st.metric("Confidence", f"{conf_pct}%")

    st.markdown(f"""
    <div class='confidence-bar-bg'>
        <div class='confidence-bar-fill' style='width:{conf_pct}%;'></div>
    </div>
    <div style='color:#64748b; font-size:12px; margin-top:4px;'>Model confidence: {conf_pct}%</div>
    """, unsafe_allow_html=True)

    # Top predictions
    if len(result.top_predictions) > 1:
        st.markdown("**Alternative diagnoses:**")
        alt_cols = st.columns(len(result.top_predictions))
        for col, (dis, prob) in zip(alt_cols, result.top_predictions):
            col.metric(dis, f"{int(prob*100)}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Safety notes ──────────────────────────────────────────────────────
    st.markdown("### 🛡️ Safety Notes")
    for note in result.safety_notes:
        st.markdown(f"<div class='safety-note'>{note}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Medicine cards ────────────────────────────────────────────────────
    st.markdown("### 💊 Recommended Medicines")

    safety_color = {"High": "pill-green", "Moderate": "pill-yellow", "Low": "pill-red"}

    for med in result.recommendations:
        sc  = safety_color.get(med.safety_profile, "pill-blue")
        eff = int(med.effectiveness_score * 100)
        rel = int(med.relevance_score * 100)

        st.markdown(f"""
        <div class='med-card'>
            <span class='rank-badge'>{med.rank}</span>
            <span class='med-name'>{med.name}</span>
            <p class='med-desc'>{med.description}</p>
            <span class='pill pill-blue'>💉 {med.dosage}</span>
            <span class='pill {sc}'>🛡 Safety: {med.safety_profile}</span>
            <span class='pill pill-green'>✅ Effectiveness: {eff}%</span>
            <span class='pill pill-blue'>🎯 Relevance: {rel}%</span>
            <details style='margin-top:10px;'>
                <summary style='color:#94a3b8; cursor:pointer; font-size:13px;'>⚠️ Warnings & Precautions</summary>
                <div style='color:#fca5a5; font-size:13px; margin-top:6px; padding-left:12px;'>{med.warnings}</div>
            </details>
        </div>
        """, unsafe_allow_html=True)

    # Download summary
    summary_lines = [
        f"MediRec AI — Recommendation Report",
        f"Predicted Disease: {result.predicted_disease}",
        f"Confidence: {conf_pct}%",
        f"Patient Age: {profile.age} | Gender: {profile.gender}",
        f"Symptoms: {', '.join(profile.symptoms)}",
        "",
        "MEDICINE RECOMMENDATIONS",
    ]
    for med in result.recommendations:
        summary_lines += [
            f"[{med.rank}] {med.name}",
            f"    Dosage: {med.dosage}",
            f"    Warnings: {med.warnings}",
            "",
        ]
    st.download_button(
        "📥 Download Report",
        data="\n".join(summary_lines),
        file_name="medirec_report.txt",
        mime="text/plain",
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊  Model Performance":
    st.markdown("## 📊 Model Performance Dashboard")

    df = load_metrics_df()
    if df is None:
        st.warning("No metrics found. Run `python train.py` first.")
        st.stop()

    # ── Summary table ─────────────────────────────────────────────────────
    st.markdown("### 🏆 Comparison Table")
    st.dataframe(
        df.style
          .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score"],
                         color="#064e3b")
          .format({"Accuracy": "{:.4f}", "Precision": "{:.4f}",
                   "Recall":   "{:.4f}", "F1-Score":  "{:.4f}"}),
        use_container_width=True,
    )

    # ── Bar charts ────────────────────────────────────────────────────────
    st.markdown("### 📈 Accuracy & F1-Score Comparison")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f172a")
    palette = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#06b6d4", "#ec4899"]

    for ax, col in zip(axes, ["Accuracy", "F1-Score"]):
        sorted_df = df.sort_values(col, ascending=True)
        bars = ax.barh(sorted_df["Model"], sorted_df[col], color=palette[:len(sorted_df)], height=0.55)
        ax.set_facecolor("#1e293b")
        ax.set_xlabel(col, color="#94a3b8")
        ax.set_title(col, color="#e2e8f0", fontsize=13, fontweight="bold")
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#334155")
        ax.set_xlim(0, 1.05)
        ax.bar_label(bars, fmt="%.4f", padding=3, color="#e2e8f0", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── ANN training history ──────────────────────────────────────────────
    ann_hist_path = os.path.join(MODEL_DIR, "ann_history.json")
    if os.path.exists(ann_hist_path):
        st.markdown("### 🧠 ANN Training History")
        with open(ann_hist_path) as f:
            hist = json.load(f)

        epochs = list(range(1, len(hist["loss"]) + 1))
        fig2, ax2 = plt.subplots(1, 2, figsize=(14, 4))
        fig2.patch.set_facecolor("#0f172a")

        for ax, (t_key, v_key, title, t_color, v_color) in zip(ax2, [
            ("loss", "val_loss", "Loss", "#3b82f6", "#f59e0b"),
            ("accuracy", "val_accuracy", "Accuracy", "#10b981", "#ec4899"),
        ]):
            ax.plot(epochs, hist[t_key], color=t_color, lw=2, label=f"Train {title}")
            ax.plot(epochs, hist[v_key], color=v_color, lw=2, linestyle="--", label=f"Val {title}")
            ax.set_facecolor("#1e293b")
            ax.set_title(title, color="#e2e8f0", fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch", color="#94a3b8")
            ax.tick_params(colors="#94a3b8")
            ax.spines[:].set_color("#334155")
            ax.legend(facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Saved chart images ────────────────────────────────────────────────
    for img_name, caption in [
        ("confusion_matrix.png", "Confusion Matrix — Best ML Model"),
        ("roc_curves.png",       "ROC Curves"),
    ]:
        path = os.path.join(ASSETS_DIR, img_name)
        if os.path.exists(path):
            st.markdown(f"### 🗂️ {caption}")
            st.image(path, use_column_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    st.markdown("## ℹ️ About This Project")

    st.markdown("""
    <div class='step-card' style='border-color:#8b5cf6;'>
        <div class='step-num'>PROJECT ABSTRACT</div>
        <div class='step-text'>
            MediRec AI is a final-year project that demonstrates the application of Machine Learning
            and Deep Learning to healthcare. Given a patient's symptoms and profile, the system
            predicts the most probable disease and recommends medicines ranked by effectiveness
            and patient-specific safety.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Tools & Technologies")
    tools = {
        "Python 3.11+":        "Core programming language",
        "Scikit-learn":        "ML models (RF, SVM, KNN, NB, DT) + preprocessing",
        "TensorFlow / Keras":  "Deep learning (ANN, LSTM)",
        "Pandas / NumPy":      "Data manipulation & numerical computing",
        "Streamlit":           "Web application framework",
        "Matplotlib / Seaborn":"Data visualisation",
        "Joblib":              "Model serialisation",
        "Google Colab":        "Training notebook environment",
    }
    for tool, desc in tools.items():
        st.markdown(f"""
        <div class='step-card'>
            <div class='step-num' style='color:#8b5cf6;'>{tool}</div>
            <div class='step-text'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📁 Project Structure")
    st.code("""
medicine_recommendation_system/
├── dataset/
│   ├── generate_dataset.py   ← Synthetic dataset generator
│   ├── symptoms_diseases.csv ← Training dataset
│   └── medicine_data.csv     ← Medicine → disease mapping
├── models/                   ← Saved models & artefacts
├── assets/                   ← Generated plots
├── notebooks/
│   └── training.ipynb        ← Google Colab notebook
├── app.py                    ← Streamlit UI (this file)
├── recommendation.py         ← Recommendation engine
├── preprocess.py             ← Data cleaning & encoding
├── train.py                  ← Model training script
├── evaluate.py               ← Evaluation & plots
├── requirements.txt
└── README.md
    """, language="")

    st.markdown("### ⚠️ Disclaimer")
    st.error("This system is strictly for **educational and research purposes**. It does not replace professional medical advice, diagnosis, or treatment. Always consult a licensed healthcare provider.")
