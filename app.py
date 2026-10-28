import streamlit as st
from modules.technician import technician_dashboard
from modules.doctor import doctor_dashboard

st.set_page_config(
    page_title="LabInsight AI - Medical Data Analysis & Disease Prediction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom header styling
st.title("🧪 LabInsight AI")
st.caption("Advanced Medical Laboratory Data Analysis & Explainable AI Disease Prediction Platform")

# Sidebar navigation & status
st.sidebar.markdown("## 🏥 Navigation")
page = st.sidebar.radio(
    "Select Role Dashboard",
    ["Lab Technician", "Doctor"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Status")
st.sidebar.success("✅ SQLite Database Connected")
st.sidebar.success("✅ 3 ML Models Loaded (Diabetes, Kidney, Heart)")
st.sidebar.success("✅ SHAP Explainability Engine Active")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Project:** Final-Year Engineering Project  \n"
    "**Domain:** Healthcare Analytics & Machine Learning  \n"
    "**Version:** 2.0 (Production-Ready)  \n"
)

st.sidebar.caption("⚠️ For Educational & Research Demonstration Only")

# Route to active dashboard
if page == "Lab Technician":
    technician_dashboard()
else:
    doctor_dashboard()