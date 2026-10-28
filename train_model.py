import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "diabetes_model.pkl"


# ==========================================================
# LOAD DATASET
# ==========================================================

print("Loading diabetes dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))


# ==========================================================
# FEATURES AND TARGET
# ==========================================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================================
# MACHINE LEARNING PIPELINE
# ==========================================================

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    ("scaler", StandardScaler()),

    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])


# ==========================================================
# TRAIN MODEL
# ==========================================================

print("\nTraining diabetes model...")

model.fit(X_train, y_train)


# ==========================================================
# EVALUATE MODEL
# ==========================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(model, MODEL_PATH)

print("\n✅ Diabetes model saved successfully!")
print("Model path:", MODEL_PATH)