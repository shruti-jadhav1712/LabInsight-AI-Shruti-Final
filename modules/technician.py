"""
LabInsight AI - Lab Technician Dashboard
Allows laboratory technicians to register or look up patients, enter laboratory
test results for Diabetes, Kidney, and Heart panels, run ML predictions,
generate SHAP explainability insights, save reports, and generate PDF summaries.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from modules.database import (
    add_patient,
    add_lab_report,
    get_patient,
    get_all_patients,
    suggest_next_patient_id,
    update_diabetes_prediction,
    update_kidney_prediction,
    update_heart_prediction,
)
from modules.prediction import (
    predict_diabetes,
    predict_kidney,
    predict_heart,
    diabetes_model,
)
from modules.shap_explainer import explain_diabetes
from modules.report_generator import generate_pdf_report
from modules.utils import (
    REFERENCE_RANGES,
    validate_patient_id,
    validate_patient_name,
)


def technician_dashboard():
    st.header("🧪 Lab Technician Dashboard")
    st.caption("Enter patient details, record clinical laboratory measurements, run AI predictions, and generate reports.")

    # Initialize session state keys
    for key in [
        "diabetes_result", "diabetes_confidence",
        "kidney_result", "kidney_confidence",
        "heart_result", "heart_confidence",
        "last_saved_report_id", "last_saved_patient_id",
        "technician_shap_result",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None

    # --------------------------------------------------------
    # 1. PATIENT MANAGEMENT & LOOKUP
    # --------------------------------------------------------
    st.subheader("👤 Patient Identification & Registration")

    patient_mode = st.radio(
        "Patient Mode",
        ["📝 New Patient Registration", "🔍 Look Up Existing Patient"],
        horizontal=True,
        key="technician_patient_mode",
    )

    suggested_id = suggest_next_patient_id()

    # Pre-filled default values
    default_id = suggested_id
    default_name = ""
    default_age = 30
    default_phone = ""
    default_gender = "Select Gender"

    if patient_mode == "🔍 Look Up Existing Patient":
        existing_patients = get_all_patients()
        if existing_patients:
            patient_options = {
                f"{p['patient_name']} (ID: {p['patient_id']}, Age: {p['age'] or 'N/A'})": p
                for p in existing_patients
            }
            selected_patient_label = st.selectbox(
                "Select Existing Patient",
                list(patient_options.keys()),
                key="technician_patient_selector",
            )
            selected_record = patient_options[selected_patient_label]

            default_id = str(selected_record["patient_id"])
            default_name = str(selected_record["patient_name"])
            default_age = int(selected_record["age"]) if selected_record["age"] is not None else 30
            default_phone = str(selected_record["phone"] or "")
            default_gender = str(selected_record["gender"] or "Select Gender")
            if default_gender not in ["Male", "Female", "Other"]:
                default_gender = "Select Gender"

            st.success(f"Loaded Patient: **{default_name}** (`{default_id}`)")
        else:
            st.info("No registered patients found in database yet. Please register a new patient.")

    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input(
            "Patient ID *",
            value=default_id,
            placeholder="e.g. P001",
            help="Unique identifier for the patient record.",
            key="tech_patient_id_input",
        )
    with col2:
        patient_name = st.text_input(
            "Patient Name *",
            value=default_name,
            placeholder="Enter full name",
            help="Full legal or clinical name of the patient.",
            key="tech_patient_name_input",
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input(
            "Age (years) *",
            min_value=1,
            max_value=120,
            value=default_age,
            help="Patient age. Required for Diabetes, Kidney, and Heart risk models.",
            key="tech_age_input",
        )
    with col2:
        phone_number = st.text_input(
            "Phone Number",
            value=default_phone,
            placeholder="Contact number",
            key="tech_phone_input",
        )
    with col3:
        gender_index = 0
        if default_gender == "Male":
            gender_index = 1
        elif default_gender == "Female":
            gender_index = 2
        elif default_gender == "Other":
            gender_index = 3

        gender = st.selectbox(
            "Gender",
            ["Select Gender", "Male", "Female", "Other"],
            index=gender_index,
            key="tech_gender_input",
        )

    # --------------------------------------------------------
    # 2. LABORATORY TEST PANELS
    # --------------------------------------------------------
    st.divider()
    st.subheader("🧪 Laboratory Test Parameter Entry")
    st.caption("Enter laboratory values for one or more diagnostic panels. Required model parameters are marked with *.")

    # --------------------------------------------------------
    # PANEL A: DIABETES TEST
    # --------------------------------------------------------
    with st.expander("🩸 Diabetes Laboratory Panel (8 Required Features)", expanded=False):
        st.markdown("**Parameters Required for Diabetes Prediction:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Pedigree, Age")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pregnancies = st.number_input(
                "Pregnancies *", 0.0, 20.0, 0.0, step=1.0,
                help="Number of times pregnant (0 for males or nulliparous)",
                key="diab_pregnancies",
            )
        with col2:
            glucose = st.number_input(
                "Glucose (mg/dL) *", 0.0, 300.0, 120.0, step=1.0,
                help=f"Reference range: {REFERENCE_RANGES['glucose']['min']} - {REFERENCE_RANGES['glucose']['max']} mg/dL (fasting)",
                key="diab_glucose",
            )
        with col3:
            blood_pressure = st.number_input(
                "Blood Pressure (mmHg) *", 0.0, 200.0, 70.0, step=1.0,
                help=f"Diastolic blood pressure. Reference: {REFERENCE_RANGES['blood_pressure']['min']} - {REFERENCE_RANGES['blood_pressure']['max']} mmHg",
                key="diab_bp",
            )
        with col4:
            skin_thickness = st.number_input(
                "Skin Thickness (mm) *", 0.0, 100.0, 20.0, step=1.0,
                help=f"Triceps skin fold thickness. Reference: {REFERENCE_RANGES['skin_thickness']['min']} - {REFERENCE_RANGES['skin_thickness']['max']} mm",
                key="diab_skin",
            )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            insulin = st.number_input(
                "Insulin (µIU/mL) *", 0.0, 900.0, 80.0, step=1.0,
                help=f"2-Hour serum insulin. Reference: {REFERENCE_RANGES['insulin']['min']} - {REFERENCE_RANGES['insulin']['max']} µIU/mL",
                key="diab_insulin",
            )
        with col2:
            bmi = st.number_input(
                "BMI (kg/m²) *", 0.0, 70.0, 25.0, step=0.1,
                help=f"Body mass index. Normal: {REFERENCE_RANGES['bmi']['min']} - {REFERENCE_RANGES['bmi']['max']}",
                key="diab_bmi",
            )
        with col3:
            diabetes_pedigree = st.number_input(
                "Pedigree Function *", 0.0, 3.0, 0.5, step=0.01,
                help="Diabetes pedigree genetic risk function score",
                key="diab_pedigree",
            )
        with col4:
            st.write("")
            st.write("")
            if st.button("🔍 Predict Diabetes Risk", key="btn_pred_diabetes", use_container_width=True):
                try:
                    result, confidence = predict_diabetes(
                        pregnancies=pregnancies,
                        glucose=glucose,
                        blood_pressure=blood_pressure,
                        skin_thickness=skin_thickness,
                        insulin=insulin,
                        bmi=bmi,
                        diabetes_pedigree=diabetes_pedigree,
                        age=age,
                    )
                    st.session_state.diabetes_result = result
                    st.session_state.diabetes_confidence = confidence
                    st.success(f"{result} ({confidence:.2f}% confidence)")
                except Exception as e:
                    st.error(f"Diabetes prediction error: {e}")

        if st.session_state.diabetes_result is not None:
            st.info(f"**Current Result:** {st.session_state.diabetes_result} | **Confidence:** {st.session_state.diabetes_confidence:.2f}%")

        # Diabetes SHAP Explainer
        st.markdown("---")
        st.markdown("#### 🔬 Explain Diabetes Model Output (SHAP)")
        if st.button("Generate SHAP Feature Contributions", key="btn_tech_shap", use_container_width=True):
            try:
                diabetes_input = pd.DataFrame([{
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": blood_pressure,
                    "SkinThickness": skin_thickness,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": diabetes_pedigree,
                    "Age": age,
                }])
                shap_exp = explain_diabetes(diabetes_model, diabetes_input)
                st.session_state.technician_shap_result = shap_exp
                st.success("SHAP Explanation generated successfully.")
            except Exception as e:
                st.error(f"SHAP explanation error: {e}")

        if st.session_state.technician_shap_result is not None:
            st.dataframe(st.session_state.technician_shap_result, use_container_width=True, hide_index=True)
            st.caption("Positive SHAP values push prediction towards Risk; negative values push towards No Risk.")

    # --------------------------------------------------------
    # PANEL B: KIDNEY DISEASE TEST
    # --------------------------------------------------------
    with st.expander("🫘 Chronic Kidney Disease Laboratory Panel (24 Parameters)", expanded=False):
        st.markdown("**Key Kidney Biomarkers:** Specific Gravity, Albumin, Sugar, Blood Urea, Creatinine, Electrolytes, Hemoglobin")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            specific_gravity = st.number_input(
                "Specific Gravity *", 1.000, 1.050, 1.020, step=0.001, format="%.3f",
                key="kid_sg",
            )
        with col2:
            albumin = st.number_input(
                "Albumin (0-5) *", 0.0, 5.0, 0.0, step=1.0, key="kid_al"
            )
        with col3:
            sugar = st.number_input(
                "Sugar (0-5) *", 0.0, 5.0, 0.0, step=1.0, key="kid_su"
            )
        with col4:
            blood_glucose = st.number_input(
                "Blood Glucose (mg/dL) *", 0.0, 500.0, 120.0, step=1.0, key="kid_bgr"
            )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            blood_urea = st.number_input(
                "Blood Urea (mg/dL) *", 0.0, 400.0, 40.0, step=1.0, key="kid_bu"
            )
        with col2:
            creatinine = st.number_input(
                "Creatinine (mg/dL) *", 0.0, 20.0, 1.0, step=0.1, key="kid_sc"
            )
        with col3:
            sodium = st.number_input(
                "Sodium (mEq/L) *", 0.0, 200.0, 140.0, step=1.0, key="kid_sod"
            )
        with col4:
            potassium = st.number_input(
                "Potassium (mEq/L) *", 0.0, 15.0, 4.5, step=0.1, key="kid_pot"
            )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            hemoglobin = st.number_input(
                "Hemoglobin (g/dL) *", 0.0, 30.0, 14.0, step=0.1, key="kid_hemo"
            )
        with col2:
            packed_cell_volume = st.number_input(
                "Packed Cell Volume (%) *", 0.0, 70.0, 40.0, step=1.0, key="kid_pcv"
            )
        with col3:
            white_blood_cell_count = st.number_input(
                "WBC Count (/µL) *", 0.0, 30000.0, 8000.0, step=100.0, key="kid_wc"
            )
        with col4:
            red_blood_cell_count = st.number_input(
                "RBC Count (M/µL) *", 0.0, 10.0, 5.0, step=0.1, key="kid_rc"
            )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            red_blood_cells = st.selectbox(
                "Red Blood Cells", ["normal", "abnormal"], key="kid_rbc"
            )
        with col2:
            pus_cells = st.selectbox(
                "Pus Cells", ["normal", "abnormal"], key="kid_pc"
            )
        with col3:
            pus_cell_clumps = st.selectbox(
                "Pus Cell Clumps", ["notpresent", "present"], key="kid_pcc"
            )
        with col4:
            bacteria = st.selectbox(
                "Bacteria", ["notpresent", "present"], key="kid_ba"
            )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            hypertension = st.selectbox(
                "Hypertension", ["no", "yes"], key="kid_htn"
            )
        with col2:
            diabetes_mellitus = st.selectbox(
                "Diabetes Mellitus", ["no", "yes"], key="kid_dm"
            )
        with col3:
            coronary_artery_disease = st.selectbox(
                "Coronary Artery Disease", ["no", "yes"], key="kid_cad"
            )
        with col4:
            appetite = st.selectbox(
                "Appetite", ["good", "poor"], key="kid_appet"
            )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pedal_edema = st.selectbox(
                "Pedal Edema", ["no", "yes"], key="kid_pe"
            )
        with col2:
            anemia = st.selectbox(
                "Anemia", ["no", "yes"], key="kid_ane"
            )
        with col3:
            st.write("")
            st.write("")
            if st.button("🔍 Predict Kidney Risk", key="btn_pred_kidney", use_container_width=True):
                try:
                    result, confidence = predict_kidney(
                        age=age,
                        blood_pressure=blood_pressure,
                        specific_gravity=specific_gravity,
                        albumin=albumin,
                        sugar=sugar,
                        red_blood_cells=red_blood_cells,
                        pus_cells=pus_cells,
                        pus_cell_clumps=pus_cell_clumps,
                        bacteria=bacteria,
                        blood_glucose=blood_glucose,
                        blood_urea=blood_urea,
                        creatinine=creatinine,
                        sodium=sodium,
                        potassium=potassium,
                        hemoglobin=hemoglobin,
                        packed_cell_volume=packed_cell_volume,
                        white_blood_cell_count=white_blood_cell_count,
                        red_blood_cell_count=red_blood_cell_count,
                        hypertension=hypertension,
                        diabetes_mellitus=diabetes_mellitus,
                        coronary_artery_disease=coronary_artery_disease,
                        appetite=appetite,
                        pedal_edema=pedal_edema,
                        anemia=anemia,
                    )
                    st.session_state.kidney_result = result
                    st.session_state.kidney_confidence = confidence
                    st.success(f"{result} ({confidence:.2f}% confidence)")
                except Exception as e:
                    st.error(f"Kidney prediction error: {e}")
        with col4:
            pass

        if st.session_state.kidney_result is not None:
            st.info(f"**Current Result:** {st.session_state.kidney_result} | **Confidence:** {st.session_state.kidney_confidence:.2f}%")

    # --------------------------------------------------------
    # PANEL C: HEART DISEASE TEST
    # --------------------------------------------------------
    with st.expander("❤️ Heart Disease Laboratory Panel (13 Parameters)", expanded=False):
        st.markdown("**Key Cardiovascular Indicators:** Chest Pain Type, Resting BP, Cholesterol, ECG, Max Heart Rate, ST Depression")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            heart_sex_val = 1 if gender == "Male" else 0
            sex = st.selectbox(
                "Biological Sex *", [1, 0],
                index=0 if heart_sex_val == 1 else 1,
                format_func=lambda x: "Male (1)" if x == 1 else "Female (0)",
                key="heart_sex_input",
            )
        with col2:
            chest_pain_type = st.selectbox(
                "Chest Pain Type (0-3) *", [0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0: Typical Angina",
                    1: "1: Atypical Angina",
                    2: "2: Non-anginal Pain",
                    3: "3: Asymptomatic",
                }.get(x, str(x)),
                key="heart_cp_input",
            )
        with col3:
            resting_blood_pressure = st.number_input(
                "Resting Blood Pressure (mmHg) *", 0.0, 250.0, 120.0, step=1.0,
                key="heart_trestbps_input",
            )
        with col4:
            cholesterol = st.number_input(
                "Serum Cholesterol (mg/dL) *", 0.0, 700.0, 200.0, step=1.0,
                key="heart_chol_input",
            )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            fasting_blood_sugar = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL *", [0, 1],
                format_func=lambda x: "Yes (> 120)" if x == 1 else "No (≤ 120)",
                key="heart_fbs_input",
            )
        with col2:
            resting_ecg = st.selectbox(
                "Resting ECG (0-2) *", [0, 1, 2],
                format_func=lambda x: {
                    0: "0: Normal",
                    1: "1: ST-T Abnormality",
                    2: "2: LV Hypertrophy",
                }.get(x, str(x)),
                key="heart_restecg_input",
            )
        with col3:
            max_heart_rate = st.number_input(
                "Max Heart Rate Achieved (bpm) *", 0.0, 250.0, 150.0, step=1.0,
                key="heart_thalach_input",
            )
        with col4:
            exercise_angina = st.selectbox(
                "Exercise Induced Angina *", [0, 1],
                format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
                key="heart_exang_input",
            )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            oldpeak = st.number_input(
                "ST Depression (Oldpeak) *", 0.0, 10.0, 1.0, step=0.1,
                key="heart_oldpeak_input",
            )
        with col2:
            slope = st.selectbox(
                "Slope of Peak Exercise ST *", [0, 1, 2],
                format_func=lambda x: {
                    0: "0: Upsloping",
                    1: "1: Flat",
                    2: "2: Downsloping",
                }.get(x, str(x)),
                key="heart_slope_input",
            )
        with col3:
            ca = st.selectbox(
                "Major Vessels Colored (0-4) *", [0, 1, 2, 3, 4],
                key="heart_ca_input",
            )
        with col4:
            thal = st.selectbox(
                "Thalassemia (0-3) *", [0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0: Null / Unknown",
                    1: "1: Normal",
                    2: "2: Fixed Defect",
                    3: "3: Reversible Defect",
                }.get(x, str(x)),
                key="heart_thal_input",
            )

        if st.button("🔍 Predict Heart Disease Risk", key="btn_pred_heart", use_container_width=True):
            try:
                result, confidence = predict_heart(
                    age=age,
                    sex=sex,
                    chest_pain_type=chest_pain_type,
                    resting_blood_pressure=resting_blood_pressure,
                    cholesterol=cholesterol,
                    fasting_blood_sugar=fasting_blood_sugar,
                    resting_ecg=resting_ecg,
                    max_heart_rate=max_heart_rate,
                    exercise_angina=exercise_angina,
                    oldpeak=oldpeak,
                    slope=slope,
                    ca=ca,
                    thal=thal,
                )
                st.session_state.heart_result = result
                st.session_state.heart_confidence = confidence
                st.success(f"{result} ({confidence:.2f}% confidence)")
            except Exception as e:
                st.error(f"Heart prediction error: {e}")

        if st.session_state.heart_result is not None:
            st.info(f"**Current Result:** {st.session_state.heart_result} | **Confidence:** {st.session_state.heart_confidence:.2f}%")

    # --------------------------------------------------------
    # 3. AI PREDICTIONS OVERVIEW & BATCH RUN
    # --------------------------------------------------------
    st.divider()
    st.subheader("📊 AI Disease Risk Overview")

    if st.button("🚀 Run All 3 Disease Risk Predictions", key="btn_run_all_predictions", use_container_width=True):
        errors = []
        try:
            res_d, conf_d = predict_diabetes(
                pregnancies=pregnancies,
                glucose=glucose,
                blood_pressure=blood_pressure,
                skin_thickness=skin_thickness,
                insulin=insulin,
                bmi=bmi,
                diabetes_pedigree=diabetes_pedigree,
                age=age,
            )
            st.session_state.diabetes_result = res_d
            st.session_state.diabetes_confidence = conf_d
        except Exception as e:
            errors.append(f"Diabetes: {e}")

        try:
            res_k, conf_k = predict_kidney(
                age=age,
                blood_pressure=blood_pressure,
                specific_gravity=specific_gravity,
                albumin=albumin,
                sugar=sugar,
                red_blood_cells=red_blood_cells,
                pus_cells=pus_cells,
                pus_cell_clumps=pus_cell_clumps,
                bacteria=bacteria,
                blood_glucose=blood_glucose,
                blood_urea=blood_urea,
                creatinine=creatinine,
                sodium=sodium,
                potassium=potassium,
                hemoglobin=hemoglobin,
                packed_cell_volume=packed_cell_volume,
                white_blood_cell_count=white_blood_cell_count,
                red_blood_cell_count=red_blood_cell_count,
                hypertension=hypertension,
                diabetes_mellitus=diabetes_mellitus,
                coronary_artery_disease=coronary_artery_disease,
                appetite=appetite,
                pedal_edema=pedal_edema,
                anemia=anemia,
            )
            st.session_state.kidney_result = res_k
            st.session_state.kidney_confidence = conf_k
        except Exception as e:
            errors.append(f"Kidney: {e}")

        try:
            res_h, conf_h = predict_heart(
                age=age,
                sex=sex,
                chest_pain_type=chest_pain_type,
                resting_blood_pressure=resting_blood_pressure,
                cholesterol=cholesterol,
                fasting_blood_sugar=fasting_blood_sugar,
                resting_ecg=resting_ecg,
                max_heart_rate=max_heart_rate,
                exercise_angina=exercise_angina,
                oldpeak=oldpeak,
                slope=slope,
                ca=ca,
                thal=thal,
            )
            st.session_state.heart_result = res_h
            st.session_state.heart_confidence = conf_h
        except Exception as e:
            errors.append(f"Heart: {e}")

        if errors:
            st.warning("Completed with warnings:\n" + "\n".join(errors))
        else:
            st.success("All 3 disease models evaluated successfully!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🩸 Diabetes")
        if st.session_state.diabetes_result is not None:
            if "No" in st.session_state.diabetes_result:
                st.success(st.session_state.diabetes_result)
            else:
                st.error(st.session_state.diabetes_result)
            st.write(f"Confidence: **{st.session_state.diabetes_confidence:.2f}%**")
        else:
            st.info("Not evaluated yet")

    with col2:
        st.markdown("### 🫘 Kidney Disease")
        if st.session_state.kidney_result is not None:
            if "No" in st.session_state.kidney_result:
                st.success(st.session_state.kidney_result)
            else:
                st.error(st.session_state.kidney_result)
            st.write(f"Confidence: **{st.session_state.kidney_confidence:.2f}%**")
        else:
            st.info("Not evaluated yet")

    with col3:
        st.markdown("### ❤️ Heart Disease")
        if st.session_state.heart_result is not None:
            if "No" in st.session_state.heart_result:
                st.success(st.session_state.heart_result)
            else:
                st.error(st.session_state.heart_result)
            st.write(f"Confidence: **{st.session_state.heart_confidence:.2f}%**")
        else:
            st.info("Not evaluated yet")

    # --------------------------------------------------------
    # 4. SAVE COMPLETE LABORATORY REPORT
    # --------------------------------------------------------
    st.divider()
    st.subheader("💾 Commit & Persist Laboratory Record")

    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        save_clicked = st.button("💾 Save Complete Laboratory Report to Database", type="primary", use_container_width=True)
    with col_btn2:
        reset_clicked = st.button("🔄 Reset Form & Start New Report", use_container_width=True)

    if reset_clicked:
        for k in ["diabetes_result", "diabetes_confidence", "kidney_result", "kidney_confidence", "heart_result", "heart_confidence", "last_saved_report_id", "technician_shap_result"]:
            st.session_state[k] = None
        st.rerun()

    if save_clicked:
        # Validation
        valid_id, id_msg = validate_patient_id(patient_id)
        if not valid_id:
            st.error(id_msg)
            return

        valid_name, name_msg = validate_patient_name(patient_name)
        if not valid_name:
            st.error(name_msg)
            return

        gender_val = None if gender == "Select Gender" else gender

        try:
            # 1. Add / Update Patient record
            add_patient(
                patient_id=patient_id,
                name=patient_name,
                age=age,
                phone_number=phone_number,
                gender=gender_val,
            )

            # 2. Add Laboratory Report atomically with current predictions
            report_id = add_lab_report(
                patient_id=patient_id,
                # Common
                glucose=glucose,
                blood_pressure=blood_pressure,
                cholesterol=cholesterol,
                creatinine=creatinine,
                blood_urea=blood_urea,
                hemoglobin=hemoglobin,
                bmi=bmi,
                # Diabetes
                pregnancies=pregnancies,
                skin_thickness=skin_thickness,
                insulin=insulin,
                diabetes_pedigree=diabetes_pedigree,
                # Kidney
                specific_gravity=specific_gravity,
                albumin=albumin,
                sugar=sugar,
                red_blood_cells=red_blood_cells,
                pus_cells=pus_cells,
                pus_cell_clumps=pus_cell_clumps,
                bacteria=bacteria,
                blood_glucose=blood_glucose,
                sodium=sodium,
                potassium=potassium,
                packed_cell_volume=packed_cell_volume,
                white_blood_cell_count=white_blood_cell_count,
                red_blood_cell_count=red_blood_cell_count,
                hypertension=hypertension,
                diabetes_mellitus=diabetes_mellitus,
                coronary_artery_disease=coronary_artery_disease,
                appetite=appetite,
                pedal_edema=pedal_edema,
                anemia=anemia,
                # Heart
                sex=sex,
                chest_pain_type=chest_pain_type,
                resting_blood_pressure=resting_blood_pressure,
                fasting_blood_sugar=fasting_blood_sugar,
                resting_ecg=resting_ecg,
                max_heart_rate=max_heart_rate,
                exercise_angina=exercise_angina,
                oldpeak=oldpeak,
                slope=slope,
                ca=ca,
                thal=thal,
                # Predictions (atomic)
                diabetes_prediction=st.session_state.diabetes_result,
                diabetes_confidence=st.session_state.diabetes_confidence,
                kidney_prediction=st.session_state.kidney_result,
                kidney_confidence=st.session_state.kidney_confidence,
                heart_prediction=st.session_state.heart_result,
                heart_confidence=st.session_state.heart_confidence,
            )

            st.session_state.last_saved_report_id = report_id
            st.session_state.last_saved_patient_id = patient_id

            st.success(f"✅ Laboratory Report #{report_id} successfully saved to SQLite database for Patient **{patient_name}** (`{patient_id}`)!")
            st.info("The record is now immediately available in the Doctor Dashboard.")

        except Exception as e:
            st.error(f"Error saving laboratory report: {e}")

    # --------------------------------------------------------
    # 5. VERIFICATION & PDF DOWNLOAD CARD (AFTER SAVE)
    # --------------------------------------------------------
    if st.session_state.get("last_saved_report_id"):
        saved_rep_id = st.session_state.last_saved_report_id
        saved_pid = st.session_state.last_saved_patient_id

        st.markdown("---")
        st.markdown(f"### 📋 Report #{saved_rep_id} Verification & PDF Export")

        # Compile data for PDF
        patient_obj = {
            "patient_id": saved_pid,
            "patient_name": patient_name,
            "age": age,
            "phone_number": phone_number,
            "gender": gender if gender != "Select Gender" else "N/A",
        }
        report_obj = {
            "report_id": saved_rep_id,
            "patient_id": saved_pid,
            "glucose": glucose,
            "blood_pressure": blood_pressure,
            "cholesterol": cholesterol,
            "creatinine": creatinine,
            "blood_urea": blood_urea,
            "hemoglobin": hemoglobin,
            "bmi": bmi,
            "diabetes_prediction": st.session_state.diabetes_result,
            "diabetes_confidence": st.session_state.diabetes_confidence,
            "kidney_prediction": st.session_state.kidney_result,
            "kidney_confidence": st.session_state.kidney_confidence,
            "heart_prediction": st.session_state.heart_result,
            "heart_confidence": st.session_state.heart_confidence,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            pdf_bytes = generate_pdf_report(
                patient=patient_obj,
                report_data=report_obj,
                shap_df=st.session_state.technician_shap_result,
            )

            st.download_button(
                label=f"📥 Download Report #{saved_rep_id} PDF",
                data=pdf_bytes,
                file_name=f"LabInsight_Report_{saved_pid}_{saved_rep_id}.pdf",
                mime="application/pdf",
                key="btn_download_tech_pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Could not generate PDF download: {e}")

    # --------------------------------------------------------
    # 6. DISCLAIMER
    # --------------------------------------------------------
    st.divider()
    st.caption(
        "⚠️ Disclaimer: LabInsight AI predictions are mathematical disease risk indications "
        "designed for academic research and educational demonstration. They must not be interpreted as "
        "a definitive medical diagnosis."
    )
