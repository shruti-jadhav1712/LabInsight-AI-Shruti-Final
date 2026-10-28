"""
LabInsight AI - Patient Laboratory Report Generator
Generates professional clinical laboratory and AI disease-risk reports in PDF format
using ReportLab, including patient details, lab measurements, model predictions,
SHAP feature attributions, and academic disclaimers.
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)

from modules.utils import (
    REFERENCE_RANGES,
    FEATURE_DISPLAY_NAMES,
    get_parameter_status,
    format_parameter_display,
)

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class NumberedCanvas(colors.Color):
    """Placeholder for custom canvas callbacks if needed."""
    pass


def _get_patient_dict(patient: Union[Dict, tuple, list]) -> Dict[str, Any]:
    """Normalize patient input into a standard dictionary."""
    if isinstance(patient, dict):
        return {
            "patient_id": patient.get("patient_id", "N/A"),
            "name": patient.get("patient_name") or patient.get("name", "N/A"),
            "age": patient.get("age", "N/A"),
            "phone": patient.get("phone_number") or patient.get("phone", "N/A"),
            "gender": patient.get("gender", "N/A"),
        }
    elif isinstance(patient, (tuple, list)):
        p_id = patient[0] if len(patient) > 0 else "N/A"
        name = patient[1] if len(patient) > 1 else "N/A"
        age = patient[2] if len(patient) > 2 else "N/A"
        phone = patient[3] if len(patient) > 3 else "N/A"
        gender = patient[4] if len(patient) > 4 else "N/A"
        return {
            "patient_id": p_id,
            "name": name,
            "age": age,
            "phone": phone,
            "gender": gender,
        }
    return {
        "patient_id": "N/A",
        "name": "N/A",
        "age": "N/A",
        "phone": "N/A",
        "gender": "N/A",
    }


def generate_pdf_report(
    patient: Union[Dict, tuple, list],
    report_data: Union[Dict, pd.Series],
    shap_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> bytes:
    """
    Generate a comprehensive PDF laboratory analysis report.

    Parameters
    ----------
    patient:
        Dictionary or tuple containing patient ID, name, age, phone, gender.
    report_data:
        Dictionary or Series containing laboratory test measurements and predictions.
    shap_df:
        Optional DataFrame containing SHAP feature contributions for diabetes.
    output_path:
        Optional file path to persist the generated PDF to disk.

    Returns
    -------
    bytes
        Binary content of the generated PDF document.
    """
    p_info = _get_patient_dict(patient)
    if isinstance(report_data, pd.Series):
        r_info = report_data.to_dict()
    elif isinstance(report_data, dict):
        r_info = report_data.copy()
    else:
        r_info = {}

    report_id = str(r_info.get("report_id", "N/A"))
    created_at = str(r_info.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # Prepare PDF document buffer
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styling
    primary_color = colors.HexColor("#0F3C68")    # Deep clinical navy
    secondary_color = colors.HexColor("#0284C7")  # Medical sky blue
    accent_dark = colors.HexColor("#1E293B")      # Charcoal dark
    bg_light = colors.HexColor("#F8FAFC")         # Clean slate light
    border_color = colors.HexColor("#CBD5E1")     # Neutral border

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=primary_color,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=4,
    )

    cell_bold = ParagraphStyle(
        "CellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=accent_dark,
    )

    cell_regular = ParagraphStyle(
        "CellRegular",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=accent_dark,
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#64748B"),
    )

    elements = []

    # ----------------------------------------------------
    # 1. HEADER
    # ----------------------------------------------------
    header_table_data = [
        [
            Paragraph("<b>🧪 LabInsight AI</b>", title_style),
            Paragraph(
                f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>"
                f"<b>Report ID:</b> #{report_id}",
                subtitle_style,
            ),
        ],
        [
            Paragraph("Medical Laboratory Data Analysis & Clinical Decision-Support System", subtitle_style),
            Paragraph("Confidential Medical Record", subtitle_style),
        ],
    ]

    header_table = Table(header_table_data, colWidths=[380, 160])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))
    elements.append(header_table)
    elements.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=8, spaceBefore=4))

    # ----------------------------------------------------
    # 2. PATIENT INFORMATION CARD
    # ----------------------------------------------------
    elements.append(Paragraph("Patient Identification & Report Details", section_heading))

    patient_grid = [
        [
            Paragraph("<b>Patient ID:</b>", cell_bold),
            Paragraph(str(p_info["patient_id"]), cell_regular),
            Paragraph("<b>Report Date:</b>", cell_bold),
            Paragraph(str(created_at), cell_regular),
        ],
        [
            Paragraph("<b>Patient Name:</b>", cell_bold),
            Paragraph(str(p_info["name"]), cell_regular),
            Paragraph("<b>Age:</b>", cell_bold),
            Paragraph(f"{p_info['age']} years" if p_info["age"] not in (None, "N/A") else "N/A", cell_regular),
        ],
        [
            Paragraph("<b>Gender:</b>", cell_bold),
            Paragraph(str(p_info["gender"] or "N/A"), cell_regular),
            Paragraph("<b>Contact:</b>", cell_bold),
            Paragraph(str(p_info["phone"] or "N/A"), cell_regular),
        ],
    ]

    patient_table = Table(patient_grid, colWidths=[90, 180, 90, 180])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_light),
        ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 6))

    # ----------------------------------------------------
    # 3. AI DISEASE PREDICTION SUMMARY
    # ----------------------------------------------------
    elements.append(Paragraph("AI Disease Risk Predictions", section_heading))

    def _fmt_pred(pred_val, conf_val):
        if not pred_val or pd.isna(pred_val):
            return "Not Evaluated", "N/A", colors.HexColor("#F1F5F9")
        pred_str = str(pred_val)
        conf_str = f"{float(conf_val):.1f}%" if conf_val is not None and not pd.isna(conf_val) else "N/A"
        if "Detected" in pred_str and "No" not in pred_str:
            return pred_str, conf_str, colors.HexColor("#FEE2E2")  # Warning pink/red
        else:
            return pred_str, conf_str, colors.HexColor("#DCFCE7")  # Favorable green

    d_pred, d_conf, d_bg = _fmt_pred(r_info.get("diabetes_prediction"), r_info.get("diabetes_confidence"))
    k_pred, k_conf, k_bg = _fmt_pred(r_info.get("kidney_prediction"), r_info.get("kidney_confidence"))
    h_pred, h_conf, h_bg = _fmt_pred(r_info.get("heart_prediction"), r_info.get("heart_confidence"))

    pred_table_data = [
        [
            Paragraph("<b>Target Pathology</b>", cell_bold),
            Paragraph("<b>Model Prediction</b>", cell_bold),
            Paragraph("<b>Model Confidence</b>", cell_bold),
            Paragraph("<b>Model Architecture</b>", cell_bold),
        ],
        [
            Paragraph("🩸 <b>Diabetes Mellitus</b>", cell_regular),
            Paragraph(d_pred, cell_bold),
            Paragraph(d_conf, cell_regular),
            Paragraph("Logistic Regression Pipeline (Imputer + Scaler)", cell_regular),
        ],
        [
            Paragraph("🫘 <b>Chronic Kidney Disease</b>", cell_regular),
            Paragraph(k_pred, cell_bold),
            Paragraph(k_conf, cell_regular),
            Paragraph("ColumnTransformer Pipeline (OneHot + Scaler + LR)", cell_regular),
        ],
        [
            Paragraph("❤️ <b>Heart Disease</b>", cell_regular),
            Paragraph(h_pred, cell_bold),
            Paragraph(h_conf, cell_regular),
            Paragraph("Logistic Regression Pipeline (StandardScaler + LR)", cell_regular),
        ],
    ]

    pred_table = Table(pred_table_data, colWidths=[130, 160, 90, 160])
    pred_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("BACKGROUND", (1, 1), (1, 1), d_bg),
        ("BACKGROUND", (1, 2), (1, 2), k_bg),
        ("BACKGROUND", (1, 3), (1, 3), h_bg),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(pred_table)
    elements.append(Spacer(1, 6))

    # ----------------------------------------------------
    # 4. LABORATORY TEST RESULTS TABLE
    # ----------------------------------------------------
    elements.append(Paragraph("Laboratory Test Findings & Parameter Breakdown", section_heading))

    # Selected key parameters to display
    key_params = [
        ("glucose", "Glucose (Fasting/Random)", r_info.get("glucose") or r_info.get("blood_glucose")),
        ("blood_pressure", "Blood Pressure (Systolic)", r_info.get("blood_pressure") or r_info.get("resting_blood_pressure")),
        ("cholesterol", "Serum Cholesterol", r_info.get("cholesterol")),
        ("creatinine", "Serum Creatinine", r_info.get("creatinine")),
        ("blood_urea", "Blood Urea Nitrogen (BUN)", r_info.get("blood_urea")),
        ("hemoglobin", "Hemoglobin", r_info.get("hemoglobin")),
        ("bmi", "Body Mass Index (BMI)", r_info.get("bmi")),
        ("insulin", "Serum Insulin", r_info.get("insulin")),
        ("skin_thickness", "Skin Thickness", r_info.get("skin_thickness")),
        ("diabetes_pedigree", "Diabetes Pedigree Score", r_info.get("diabetes_pedigree")),
        ("specific_gravity", "Urine Specific Gravity", r_info.get("specific_gravity")),
        ("albumin", "Urine Albumin", r_info.get("albumin")),
        ("sodium", "Serum Sodium", r_info.get("sodium")),
        ("potassium", "Serum Potassium", r_info.get("potassium")),
        ("packed_cell_volume", "Packed Cell Volume (PCV)", r_info.get("packed_cell_volume")),
        ("white_blood_cell_count", "White Blood Cell Count", r_info.get("white_blood_cell_count")),
        ("max_heart_rate", "Maximum Heart Rate", r_info.get("max_heart_rate")),
        ("oldpeak", "ST Depression (Oldpeak)", r_info.get("oldpeak")),
    ]

    lab_table_data = [
        [
            Paragraph("<b>Test Parameter</b>", cell_bold),
            Paragraph("<b>Patient Value</b>", cell_bold),
            Paragraph("<b>Standard Reference Range</b>", cell_bold),
            Paragraph("<b>Diagnostic Flag</b>", cell_bold),
        ]
    ]

    for key, display_name, val in key_params:
        if val is None or pd.isna(val) or str(val).strip() in ("", "None", "nan"):
            continue

        ref = REFERENCE_RANGES.get(key, {})
        ref_str = f"{ref.get('min', '')} - {ref.get('max', '')} {ref.get('unit', '')}" if ref else "Standard clinical"
        status_text, style_flag = get_parameter_status(key, val)

        val_str = format_parameter_display(key, val)

        flag_p = Paragraph(
            f"<font color='{'#DC2626' if status_text == 'High' else '#2563EB' if status_text == 'Low' else '#16A34A'}'><b>{status_text}</b></font>",
            cell_bold,
        )

        lab_table_data.append([
            Paragraph(display_name, cell_regular),
            Paragraph(val_str, cell_bold),
            Paragraph(ref_str, cell_regular),
            flag_p,
        ])

    # Categorical observations if present
    cat_items = [
        ("Red Blood Cells", r_info.get("red_blood_cells")),
        ("Pus Cells", r_info.get("pus_cells")),
        ("Bacteria", r_info.get("bacteria")),
        ("Hypertension", r_info.get("hypertension")),
        ("Coronary Artery Disease", r_info.get("coronary_artery_disease")),
        ("Chest Pain Type", r_info.get("chest_pain_type")),
    ]

    for cat_name, cat_val in cat_items:
        if cat_val is not None and not pd.isna(cat_val) and str(cat_val).strip() not in ("", "None", "nan"):
            lab_table_data.append([
                Paragraph(cat_name, cell_regular),
                Paragraph(str(cat_val).title(), cell_bold),
                Paragraph("Categorical finding", cell_regular),
                Paragraph("Recorded", cell_regular),
            ])

    lab_table = Table(lab_table_data, colWidths=[170, 120, 160, 90])
    lab_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#F1F5F9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, bg_light]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(lab_table)
    elements.append(Spacer(1, 6))

    # ----------------------------------------------------
    # 5. SHAP EXPLAINABILITY SUMMARY (IF AVAILABLE)
    # ----------------------------------------------------
    if shap_df is not None and not shap_df.empty:
        shap_elements = []
        shap_elements.append(Paragraph("Explainable AI (SHAP) — Diabetes Feature Attribution", section_heading))
        shap_elements.append(Paragraph(
            "SHAP (SHapley Additive exPlanations) isolates the mathematical contribution of each "
            "laboratory parameter to the diabetes classification model. Positive impacts increase predicted "
            "risk; negative impacts lower predicted risk.",
            subtitle_style,
        ))
        shap_elements.append(Spacer(1, 3))

        shap_table_data = [
            [
                Paragraph("<b>Parameter</b>", cell_bold),
                Paragraph("<b>SHAP Impact</b>", cell_bold),
                Paragraph("<b>Effect Direction</b>", cell_bold),
                Paragraph("<b>Contribution Level</b>", cell_bold),
            ]
        ]

        for _, row in shap_df.iterrows():
            param = str(row.get("Parameter", ""))
            friendly_name = FEATURE_DISPLAY_NAMES.get(param, param)
            impact = float(row.get("SHAP Impact", 0.0))
            direction = str(row.get("Direction", ""))
            abs_impact = float(row.get("Absolute Impact", abs(impact)))

            impact_color = "#16A34A" if impact >= 0 else "#DC2626"
            level_str = "High" if abs_impact > 0.5 else "Moderate" if abs_impact > 0.2 else "Low"

            shap_table_data.append([
                Paragraph(friendly_name, cell_regular),
                Paragraph(f"<font color='{impact_color}'><b>{impact:+.4f}</b></font>", cell_bold),
                Paragraph(direction, cell_regular),
                Paragraph(level_str, cell_regular),
            ])

        shap_table = Table(shap_table_data, colWidths=[170, 100, 180, 90])
        shap_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
            ("BOX", (0, 0), (-1, -1), 0.5, border_color),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#F1F5F9")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, bg_light]),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        shap_elements.append(shap_table)
        elements.append(KeepTogether(shap_elements))
        elements.append(Spacer(1, 6))

    # ----------------------------------------------------
    # 6. DISCLAIMER & LEGAL FOOTER
    # ----------------------------------------------------
    disclaimer_box = [
        [
            Paragraph(
                "<b>⚠️ MEDICAL & ACADEMIC DISCLAIMER:</b> "
                "LabInsight AI is an educational research and clinical decision-support tool. "
                "The predictions and SHAP explainability analyses presented in this document are generated by "
                "machine learning models and statistical pipelines for academic evaluation. They do NOT "
                "constitute a medical diagnosis, clinical prognosis, or treatment directive. "
                "All findings must be reviewed, verified, and interpreted by a licensed medical practitioner.",
                disclaimer_style,
            )
        ]
    ]

    disc_table = Table(disclaimer_box, colWidths=[540])
    disc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),  # Soft warm amber warning
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#F59E0B")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(KeepTogether([disc_table]))

    # Build PDF
    doc.build(elements)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes
