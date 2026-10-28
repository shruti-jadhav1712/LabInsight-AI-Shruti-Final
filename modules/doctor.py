"""
LabInsight AI - Doctor Dashboard
Enables clinicians to search patients, review comprehensive laboratory test panels,
inspect AI disease risk assessments, interpret Diabetes predictions using SHAP,
navigate report history, and download clinical summary PDF reports.
"""

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

from modules.database import (
    get_patient,
    search_patients_by_name,
    get_lab_reports,
    get_all_patients,
)
from modules.prediction import diabetes_model
from modules.shap_explainer import explain_diabetes
from modules.report_generator import generate_pdf_report
from modules.utils import (
    REFERENCE_RANGES,
    FEATURE_DISPLAY_NAMES,
    get_parameter_status,
    format_parameter_display,
)

# 8 input features required by the diabetes model pipeline in exact training order
DIABETES_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

DIABETES_COLUMN_MAP = {
    "Pregnancies": "pregnancies",
    "Glucose": "glucose",
    "BloodPressure": "blood_pressure",
    "SkinThickness": "skin_thickness",
    "Insulin": "insulin",
    "BMI": "bmi",
    "DiabetesPedigreeFunction": "diabetes_pedigree",
}


def display_parameter_table(data_dict):
    """Render a clean, formatted table for clinical parameters with status flags."""
    rows = []
    for key, val in data_dict.items():
        val_display = "N/A" if val is None or pd.isna(val) or str(val).strip() in ("", "None", "nan") else str(val)

        # Lookup reference range and flag
        ref_info = REFERENCE_RANGES.get(key.lower().replace(" ", "_"), {})
        ref_text = f"{ref_info.get('min', '')} - {ref_info.get('max', '')} {ref_info.get('unit', '')}" if ref_info else "Standard"

        status_text, style_class = get_parameter_status(key.lower().replace(" ", "_"), val)

        rows.append({
            "Parameter": key,
            "Value": val_display,
            "Reference Range": ref_text,
            "Clinical Flag": status_text,
        })

    table_df = pd.DataFrame(rows)
    st.dataframe(table_df, use_container_width=True, hide_index=True)


def doctor_dashboard():
    st.header("👨‍⚕️ Doctor Dashboard")
    st.caption("Search for a patient to view clinical laboratory reports, machine learning risk assessments, and explainable AI insights.")

    # --------------------------------------------------
    # 1. SEARCH PATIENT
    # --------------------------------------------------
    st.subheader("🔎 Patient Search")

    search_option = st.radio(
        "Search By",
        ["Patient ID", "Patient Name"],
        horizontal=True,
        key="doctor_search_option",
    )

    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        search_value = st.text_input(
            search_option,
            placeholder="e.g. P001" if search_option == "Patient ID" else "e.g. Neha",
            key="doctor_search_value",
            label_visibility="collapsed",
        )
    with col_s2:
        search_button = st.button("🔍 Search", use_container_width=True, key="doctor_search_button")

    if search_button:
        if not search_value.strip():
            st.error(f"Please enter a {search_option} to search.")
        elif search_option == "Patient ID":
            patient = get_patient(search_value.strip())
            if patient is None:
                st.session_state.pop("doctor_patient", None)
                st.session_state.pop("doctor_patient_choices", None)
                st.error(f"No patient found with ID: '{search_value.strip()}'.")
            else:
                st.session_state["doctor_patient"] = patient
                st.session_state.pop("doctor_patient_choices", None)
        else:
            patients = search_patients_by_name(search_value.strip())
            if not patients:
                st.session_state.pop("doctor_patient", None)
                st.session_state.pop("doctor_patient_choices", None)
                st.error(f"No patients found matching name: '{search_value.strip()}'.")
            elif len(patients) == 1:
                st.session_state["doctor_patient"] = patients[0]
                st.session_state.pop("doctor_patient_choices", None)
            else:
                st.session_state["doctor_patient_choices"] = patients
                st.session_state.pop("doctor_patient", None)

    # Disambiguation selector for multiple matching patients
    patient_choices = st.session_state.get("doctor_patient_choices", [])
    if patient_choices:
        st.info(f"Found {len(patient_choices)} patients matching your search. Please select:")
        patient_options = {
            f"{p[1]} | ID: {p[0]} | Age: {p[2] or 'N/A'} | Contact: {p[3] or 'N/A'}": p
            for p in patient_choices
        }
        selected_label = st.selectbox(
            "Select Matching Patient",
            list(patient_options.keys()),
            key="doctor_patient_selector",
        )
        if st.button("Confirm Patient Selection", key="confirm_doctor_patient"):
            st.session_state["doctor_patient"] = patient_options[selected_label]
            st.session_state.pop("doctor_patient_choices", None)
            st.rerun()

    # Empty state guard
    if "doctor_patient" not in st.session_state:
        st.info("👆 Enter a Patient ID or Name above to view patient diagnostic records.")
        # Quick access helper for demonstration
        all_pts = get_all_patients()
        if all_pts:
            with st.expander("📋 Registered Patients Quick Directory", expanded=False):
                quick_df = pd.DataFrame(all_pts)[["patient_id", "patient_name", "age", "gender", "phone"]]
                quick_df.columns = ["Patient ID", "Name", "Age", "Gender", "Phone"]
                st.dataframe(quick_df, use_container_width=True, hide_index=True)
        return

    patient = st.session_state["doctor_patient"]
    if patient is None or len(patient) < 5:
        st.error("Selected patient record is incomplete or corrupted.")
        return

    patient_id, patient_name, patient_age, phone_number, gender = patient

    # --------------------------------------------------
    # 2. PATIENT INFORMATION CARD
    # --------------------------------------------------
    st.divider()
    st.subheader("👤 Patient Clinical Profile")

    cols = st.columns(5)
    cols[0].metric("Patient ID", str(patient_id))
    cols[1].metric("Patient Name", str(patient_name))
    cols[2].metric("Age", f"{patient_age} yrs" if patient_age is not None else "N/A")
    cols[3].metric("Gender", str(gender or "N/A"))
    cols[4].metric("Contact", str(phone_number or "N/A"))

    # --------------------------------------------------
    # 3. LOAD LABORATORY REPORTS
    # --------------------------------------------------
    reports, column_names = get_lab_reports(patient_id)

    if not reports:
        st.warning(f"No laboratory reports found for Patient {patient_name} (`{patient_id}`).")
        st.info("Please request the Laboratory Technician to enter test results for this patient.")
        return

    df = pd.DataFrame(reports, columns=column_names)

    # Sort reports chronologically descending
    if "created_at" in df.columns:
        df["_sort_date"] = pd.to_datetime(df["created_at"], errors="coerce")
        df = df.sort_values("_sort_date", ascending=False, na_position="last").reset_index(drop=True)

    # Report Selector
    st.markdown("#### 📑 Select Laboratory Report")
    report_options = {}
    for idx, row in df.iterrows():
        rep_id = row.get("report_id", idx + 1)
        created_str = str(row.get("created_at", "Unknown Date"))
        label = f"Report #{rep_id} — {created_str}" + (" (Latest)" if idx == 0 else "")
        report_options[label] = idx

    selected_report_label = st.selectbox(
        "Active Report to Inspect",
        list(report_options.keys()),
        index=0,
        key="doctor_active_report_selector",
    )
    selected_idx = report_options[selected_report_label]
    active_report = df.iloc[selected_idx]
    report_id = str(active_report.get("report_id", "N/A"))

    # --------------------------------------------------
    # 4. LATEST / ACTIVE LABORATORY REPORT DETAILS
    # --------------------------------------------------
    st.divider()
    st.subheader(f"🧪 Laboratory Findings — Report #{report_id}")

    col1, col2 = st.columns(2)
    col1.info(f"**Report ID:** #{report_id}")
    col2.info(f"**Recorded Date:** {active_report.get('created_at', 'N/A')}")

    with st.expander("🩸 Diabetes Laboratory Panel", expanded=True):
        display_parameter_table({
            "Pregnancies": active_report.get("pregnancies"),
            "Glucose": active_report.get("glucose"),
            "Blood Pressure": active_report.get("blood_pressure"),
            "Skin Thickness": active_report.get("skin_thickness"),
            "Insulin": active_report.get("insulin"),
            "BMI": active_report.get("bmi"),
            "Diabetes Pedigree": active_report.get("diabetes_pedigree"),
            "Age": patient_age,
        })

    with st.expander("🫘 Chronic Kidney Disease Laboratory Panel", expanded=False):
        display_parameter_table({
            "Specific Gravity": active_report.get("specific_gravity"),
            "Albumin": active_report.get("albumin"),
            "Sugar": active_report.get("sugar"),
            "Blood Glucose": active_report.get("blood_glucose"),
            "Blood Urea": active_report.get("blood_urea"),
            "Creatinine": active_report.get("creatinine"),
            "Sodium": active_report.get("sodium"),
            "Potassium": active_report.get("potassium"),
            "Hemoglobin": active_report.get("hemoglobin"),
            "Packed Cell Volume": active_report.get("packed_cell_volume"),
            "White Blood Cell Count": active_report.get("white_blood_cell_count"),
            "Red Blood Cell Count": active_report.get("red_blood_cell_count"),
            "Red Blood Cells": active_report.get("red_blood_cells"),
            "Pus Cells": active_report.get("pus_cells"),
            "Pus Cell Clumps": active_report.get("pus_cell_clumps"),
            "Bacteria": active_report.get("bacteria"),
            "Hypertension": active_report.get("hypertension"),
            "Diabetes Mellitus": active_report.get("diabetes_mellitus"),
            "Coronary Artery Disease": active_report.get("coronary_artery_disease"),
            "Appetite": active_report.get("appetite"),
            "Pedal Edema": active_report.get("pedal_edema"),
            "Anemia": active_report.get("anemia"),
        })

    with st.expander("❤️ Cardiovascular Disease Laboratory Panel", expanded=False):
        display_parameter_table({
            "Biological Sex": "Male" if active_report.get("sex") == 1 else "Female" if active_report.get("sex") == 0 else active_report.get("sex"),
            "Chest Pain Type": active_report.get("chest_pain_type"),
            "Resting Blood Pressure": active_report.get("resting_blood_pressure"),
            "Cholesterol": active_report.get("cholesterol"),
            "Fasting Blood Sugar > 120": active_report.get("fasting_blood_sugar"),
            "Resting ECG": active_report.get("resting_ecg"),
            "Maximum Heart Rate": active_report.get("max_heart_rate"),
            "Exercise Angina": active_report.get("exercise_angina"),
            "Oldpeak (ST Depression)": active_report.get("oldpeak"),
            "ST Slope": active_report.get("slope"),
            "Major Vessels (CA)": active_report.get("ca"),
            "Thalassemia": active_report.get("thal"),
        })

    # --------------------------------------------------
    # 5. AI PREDICTION SUMMARY
    # --------------------------------------------------
    st.divider()
    st.subheader("🤖 AI Disease Risk Assessment Summary")

    pred_cols = st.columns(3)
    prediction_cards = [
        ("🩸 Diabetes Risk", "diabetes_prediction", "diabetes_confidence", pred_cols[0]),
        ("🫘 Kidney Disease Risk", "kidney_prediction", "kidney_confidence", pred_cols[1]),
        ("❤️ Heart Disease Risk", "heart_prediction", "heart_confidence", pred_cols[2]),
    ]

    for title, pred_k, conf_k, col in prediction_cards:
        with col:
            st.markdown(f"### {title}")
            p_val = active_report.get(pred_k)
            c_val = active_report.get(conf_k)

            if p_val is not None and not pd.isna(p_val) and str(p_val).strip() != "":
                p_str = str(p_val)
                if "No" in p_str:
                    st.success(p_str)
                else:
                    st.error(p_str)

                if c_val is not None and not pd.isna(c_val):
                    try:
                        st.write(f"Model Confidence: **{float(c_val):.2f}%**")
                    except (ValueError, TypeError):
                        st.write(f"Confidence: {c_val}")
            else:
                st.info("No prediction generated yet.")

    # --------------------------------------------------
    # 6. SHAP EXPLANATION (DIABETES)
    # --------------------------------------------------
    st.divider()
    st.subheader("🔍 Explainable AI (SHAP) — Diabetes Feature Attribution")
    st.caption("Visualizes the mathematical contribution of each laboratory biomarker to the diabetes model's classification score.")

    # Collect and validate all 8 required features for SHAP
    diabetes_input_vals = {}
    missing_features = []

    for feat_name, col_name in DIABETES_COLUMN_MAP.items():
        v = active_report.get(col_name)
        try:
            if v is None or pd.isna(v) or str(v).strip() in ("", "None", "nan"):
                missing_features.append(feat_name)
            else:
                diabetes_input_vals[feat_name] = float(v)
        except (TypeError, ValueError):
            missing_features.append(feat_name)

    try:
        if patient_age is None or pd.isna(patient_age):
            missing_features.append("Age")
        else:
            diabetes_input_vals["Age"] = float(patient_age)
    except (TypeError, ValueError):
        missing_features.append("Age")

    if missing_features:
        st.warning(f"⚠️ Missing required diabetes input features: **{', '.join(sorted(set(missing_features)))}**")
        st.caption("All 8 parameters must be recorded to calculate SHAP feature attributions.")
    else:
        st.success("✅ All 8 clinical diabetes features are available for attribution analysis.")

    # Session key tied strictly to this patient and report
    shap_state_key = f"doctor_shap_{patient_id}_{report_id}"

    # SHAP Generation Button (visible even if disabled)
    btn_disabled = bool(missing_features)
    if st.button(
        "Generate Doctor's SHAP Explanation",
        key="btn_generate_doctor_shap",
        disabled=btn_disabled,
        use_container_width=True,
    ):
        try:
            input_df = pd.DataFrame([[
                diabetes_input_vals[f] for f in DIABETES_FEATURES
            ]], columns=DIABETES_FEATURES)

            explanation = explain_diabetes(diabetes_model, input_df)

            if explanation is not None and not explanation.empty:
                st.session_state[shap_state_key] = explanation.copy()
                st.success("SHAP explanation generated successfully!")
            else:
                st.warning("SHAP explainer returned an empty result.")
        except Exception as e:
            st.error(f"SHAP explanation calculation failed: {e}")

    # Display SHAP explanation if stored in session state
    explanation = st.session_state.get(shap_state_key)

    if explanation is not None:
        st.markdown("#### 📊 Parameter Contribution Chart")

        chart_df = explanation.copy()
        chart_df["SHAP Impact"] = pd.to_numeric(chart_df["SHAP Impact"], errors="coerce").fillna(0.0)

        # Map friendly parameter labels
        chart_df["Display Name"] = chart_df["Parameter"].apply(
            lambda p: FEATURE_DISPLAY_NAMES.get(p, p)
        )

        chart_df["Contribution Direction"] = chart_df["SHAP Impact"].apply(
            lambda v: "Positive impact (Increases Risk)" if v >= 0 else "Negative impact (Lowers Risk)"
        )

        chart_df = chart_df.sort_values("SHAP Impact", key=lambda x: x.abs(), ascending=True)

        bars = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X(
                    "SHAP Impact:Q",
                    title="SHAP Impact (Log-Odds Contribution to Risk)",
                    axis=alt.Axis(format=".3f"),
                ),
                y=alt.Y(
                    "Display Name:N",
                    title="Biomarker",
                    sort=chart_df["Display Name"].tolist(),
                    axis=alt.Axis(labelLimit=300),
                ),
                color=alt.Color(
                    "Contribution Direction:N",
                    scale=alt.Scale(
                        domain=["Positive impact (Increases Risk)", "Negative impact (Lowers Risk)"],
                        range=["#16A34A", "#DC2626"],
                    ),
                    legend=alt.Legend(title="Attribution Direction"),
                ),
                tooltip=[
                    alt.Tooltip("Display Name:N", title="Biomarker"),
                    alt.Tooltip("SHAP Impact:Q", title="SHAP Value", format=".5f"),
                    alt.Tooltip("Contribution Direction:N", title="Effect"),
                ],
            )
            .properties(
                title="Diabetes Model Feature Attribution (SHAP)",
                height=350,
            )
        )

        zero_rule = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(
            color="#64748B", strokeDash=[4, 3], size=1.5
        ).encode(x="zero:Q")

        st.altair_chart(bars + zero_rule, use_container_width=True)

        st.markdown("#### 📋 Detailed SHAP Contribution Table")
        st.dataframe(
            explanation[["Parameter", "SHAP Impact", "Direction", "Absolute Impact"]],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Scientific Context: SHAP values quantify the marginal contribution of each biomarker to "
            "the logistic regression log-odds output. They explain model mathematical behavior, not clinical causation."
        )

    # --------------------------------------------------
    # 7. DOWNLOAD PATIENT REPORT (PDF)
    # --------------------------------------------------
    st.divider()
    st.subheader("📥 Export Clinical Laboratory Report")
    st.write("Generate and download a comprehensive PDF report containing all patient findings, model predictions, and SHAP explainability analyses.")

    patient_payload = {
        "patient_id": patient_id,
        "name": patient_name,
        "age": patient_age,
        "phone": phone_number,
        "gender": gender,
    }

    try:
        pdf_bytes = generate_pdf_report(
            patient=patient_payload,
            report_data=active_report.to_dict(),
            shap_df=explanation,
        )

        st.download_button(
            label=f"📄 Download Laboratory & AI Report #{report_id} (PDF)",
            data=pdf_bytes,
            file_name=f"LabInsight_Report_{patient_id}_#{report_id}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
            key="btn_download_doctor_pdf",
        )
    except Exception as e:
        st.error(f"Error compiling PDF report: {e}")

    # --------------------------------------------------
    # 8. COMPLETE LABORATORY REPORT TABLE
    # --------------------------------------------------
    st.divider()
    st.subheader("📋 Complete Laboratory Raw Records")

    exclude_cols = ["_sort_date", "prediction", "confidence"]
    clean_cols = [c for c in df.columns if c not in exclude_cols]
    st.dataframe(
        df[clean_cols].fillna("N/A"),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------
    # 9. PREVIOUS REPORT HISTORY
    # --------------------------------------------------
    st.divider()
    st.subheader("📚 Longitudinal Report History")

    if len(df) > 1:
        hist_columns = [
            "report_id", "created_at",
            "diabetes_prediction", "diabetes_confidence",
            "kidney_prediction", "kidney_confidence",
            "heart_prediction", "heart_confidence",
        ]
        avail_hist = [c for c in hist_columns if c in df.columns]
        st.dataframe(df[avail_hist].fillna("Not Evaluated"), use_container_width=True, hide_index=True)
    else:
        st.info("This is the only laboratory report recorded for this patient.")

    # --------------------------------------------------
    # 10. MEDICAL DISCLAIMER
    # --------------------------------------------------
    st.divider()
    st.caption(
        "⚠️ Medical Disclaimer: LabInsight AI is developed solely for academic and educational evaluation. "
        "Predictions and SHAP attributions represent algorithmic risk estimations and must never replace "
        "professional medical diagnosis, clinical judgment, or laboratory consultation."
    )
