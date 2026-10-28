import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import joblib


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "heart_disease.csv"

MODEL_PATH = BASE_DIR / "models" / "heart_model.pkl"


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("Columns:")
print(df.columns.tolist())


# ==========================================================
# SEPARATE FEATURES AND TARGET
# ==========================================================

X = df.drop(columns=["target"])

y = df["target"]


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# MODEL PIPELINE
# ==========================================================

model = Pipeline([
    
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        LogisticRegression(max_iter=2000)
    )
])


# ==========================================================
# TRAIN MODEL
# ==========================================================

model.fit(
    X_train,
    y_train
)


# ==========================================================
# EVALUATE MODEL
# ==========================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print("Heart Disease Model Accuracy:")
print(f"{accuracy * 100:.2f}%")

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    model,
    MODEL_PATH
)


print()
print("✅ Heart disease model saved successfully!")

print(
    f"Model path: {MODEL_PATH}"
)