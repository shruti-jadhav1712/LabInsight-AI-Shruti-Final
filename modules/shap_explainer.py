
"""""
LabInsight AI - SHAP Explanation Module

Generates feature-level explanations for the diabetes prediction model.
Compatible with a scikit-learn Pipeline containing:
    1. SimpleImputer
    2. StandardScaler
    3. LogisticRegression

SHAP values explain model output, not clinical risk or diagnosis.
"""

import numpy as np
import pandas as pd
import shap


# Features must match the order and names used during model training.
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


def _get_pipeline_steps(model):
    """
    Retrieve the preprocessing and classifier steps from the pipeline.
    """
    if not hasattr(model, "named_steps"):
        raise TypeError(
            "The diabetes model must be a scikit-learn Pipeline."
        )

    steps = model.named_steps

    if "imputer" not in steps:
        raise ValueError(
            "The model pipeline is missing the 'imputer' step."
        )

    if "scaler" not in steps:
        raise ValueError(
            "The model pipeline is missing the 'scaler' step."
        )

    if "classifier" not in steps:
        raise ValueError(
            "The model pipeline is missing the 'classifier' step."
        )

    imputer = steps["imputer"]
    scaler = steps["scaler"]
    classifier = steps["classifier"]

    if not hasattr(classifier, "coef_"):
        raise TypeError(
            "The classifier must expose coefficients, as LogisticRegression does."
        )

    return imputer, scaler, classifier


def _extract_positive_class_shap_values(shap_values, n_features):
    """
    Extract SHAP values for the positive class across SHAP output formats.
    """
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            values = np.asarray(shap_values[1])
        else:
            values = np.asarray(shap_values[0])
    else:
        values = np.asarray(shap_values)

        # Some SHAP versions return:
        # (samples, features, classes)
        if values.ndim == 3:
            values = values[:, :, -1]

    if values.ndim == 2:
        if values.shape[0] != 1:
            raise ValueError(
                "Expected one patient's input row for the explanation."
            )
        values = values[0]

    if values.ndim != 1 or len(values) != n_features:
        raise ValueError(
            "Unexpected SHAP output shape. "
            f"Received {np.asarray(shap_values).shape}; "
            f"expected {n_features} feature values."
        )

    return values.astype(float)


def explain_diabetes(model, input_data):
    """
    Explain a single patient's diabetes model output.

    Parameters
    ----------
    model:
        Trained scikit-learn diabetes Pipeline.

    input_data:
        A pandas DataFrame containing one row and the eight diabetes
        features listed in DIABETES_FEATURES.

    Returns
    -------
    pandas.DataFrame
        Columns:
            Parameter
            SHAP Impact
            Direction
            Absolute Impact

        Positive impacts increase the model's positive-class output;
        negative impacts decrease it. For LogisticRegression, these
        impacts are generally in log-odds units.
    """
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError(
            "input_data must be a pandas DataFrame."
        )

    if len(input_data) != 1:
        raise ValueError(
            "Provide exactly one patient's input row."
        )

    missing_columns = [
        feature
        for feature in DIABETES_FEATURES
        if feature not in input_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required diabetes features: "
            + ", ".join(missing_columns)
        )

    # Keep the exact feature order used during training.
    patient_data = input_data[DIABETES_FEATURES].copy()

    # Convert values to numeric and reject invalid values.
    for feature in DIABETES_FEATURES:
        patient_data[feature] = pd.to_numeric(
            patient_data[feature],
            errors="coerce",
        )

    if patient_data.isna().any().any():
        raise ValueError(
            "One or more diabetes input values are missing or invalid. "
            "Please provide all eight numeric features."
        )

    if not np.isfinite(patient_data.to_numpy(dtype=float)).all():
        raise ValueError(
            "Diabetes input values must be finite numbers."
        )

    # Extract trained pipeline components.
    imputer, scaler, classifier = _get_pipeline_steps(model)

    # Apply the same preprocessing steps used during model training.
    imputed_data = imputer.transform(patient_data)
    transformed_data = scaler.transform(imputed_data)

    # A zero reference in standardized feature space represents the
    # imputer/scaler reference point. It is not a population-average
    # clinical baseline.
    background = np.zeros(
        (1, transformed_data.shape[1]),
        dtype=float,
    )

    # Explain the fitted logistic regression classifier.
    explainer = shap.LinearExplainer(
        classifier,
        background,
    )

    shap_values = explainer.shap_values(transformed_data)

    impacts = _extract_positive_class_shap_values(
        shap_values,
        len(DIABETES_FEATURES),
    )

    explanation = pd.DataFrame({
        "Parameter": DIABETES_FEATURES,
        "SHAP Impact": impacts,
    })

    explanation["Direction"] = explanation["SHAP Impact"].apply(
        lambda value: (
            "Increases model output"
            if value > 0
            else "Decreases model output"
            if value < 0
            else "No impact"
        )
    )

    explanation["Absolute Impact"] = (
        explanation["SHAP Impact"].abs()
    )

    explanation = explanation.sort_values(
        "Absolute Impact",
        ascending=False,
    ).reset_index(drop=True)

    return explanation[
        [
            "Parameter",
            "SHAP Impact",
            "Direction",
            "Absolute Impact",
        ]
    ]


def get_diabetes_model_output(model, input_data):
    """
    Return the model's predicted class and probability for one patient.

    This is separate from SHAP so the app can display the prediction
    and its feature explanation independently.
    """
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError(
            "input_data must be a pandas DataFrame."
        )

    if len(input_data) != 1:
        raise ValueError(
            "Provide exactly one patient's input row."
        )

    missing_columns = [
        feature
        for feature in DIABETES_FEATURES
        if feature not in input_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required diabetes features: "
            + ", ".join(missing_columns)
        )

    patient_data = input_data[DIABETES_FEATURES].copy()

    for feature in DIABETES_FEATURES:
        patient_data[feature] = pd.to_numeric(
            patient_data[feature],
            errors="coerce",
        )

    if patient_data.isna().any().any():
        raise ValueError(
            "All eight diabetes input values must be numeric and present."
        )

    if not np.isfinite(patient_data.to_numpy(dtype=float)).all():
        raise ValueError(
            "Diabetes input values must be finite numbers."
        )

    predicted_class = model.predict(patient_data)[0]

    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(patient_data)[0]
        classes = list(model.classes_)

        # Prefer the class labelled 1 when available.
        if 1 in classes:
            positive_index = classes.index(1)
        else:
            positive_index = len(classes) - 1

        probability = float(probabilities[positive_index])

    return {
        "predicted_class": predicted_class,
        "positive_class_probability": probability,
    }