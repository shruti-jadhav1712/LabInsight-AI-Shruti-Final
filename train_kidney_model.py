import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------
# PATHS
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "raw" / "kidney_disease.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "kidney_model.pkl"


# -----------------------------------
# LOAD DATASET
# -----------------------------------

print("Loading kidney dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# -----------------------------------
# CLEAN COLUMN NAMES
# -----------------------------------

df.columns = df.columns.str.strip().str.lower()


# -----------------------------------
# REMOVE ID COLUMN
# -----------------------------------

if "id" in df.columns:
    df = df.drop(columns=["id"])


# -----------------------------------
# CLEAN STRING VALUES
# -----------------------------------

for column in df.select_dtypes(include="object").columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
    )


# -----------------------------------
# CLEAN MISSING VALUE SYMBOLS
# -----------------------------------

missing_values = [
    "?",
    "nan",
    "none",
    ""
]

df = df.replace(missing_values, pd.NA)


# -----------------------------------
# TARGET COLUMN
# -----------------------------------

target_column = "class"


# Remove rows where target is missing
df = df.dropna(subset=[target_column])


# -----------------------------------
# CLEAN TARGET VALUES
# -----------------------------------

df[target_column] = (
    df[target_column]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Convert:
# ckd     -> 1
# notckd  -> 0

df[target_column] = df[target_column].replace({
    "ckd": 1,
    "ckd\t": 1,
    "notckd": 0,
    "not ckd": 0
})


# Remove any unexpected target values
df = df[df[target_column].isin([0, 1])]


# -----------------------------------
# SEPARATE FEATURES AND TARGET
# -----------------------------------

X = df.drop(columns=[target_column])

y = df[target_column].astype(int)


# -----------------------------------
# IDENTIFY COLUMN TYPES
# -----------------------------------

numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumeric columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)


# -----------------------------------
# NUMERIC PIPELINE
# -----------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# -----------------------------------
# CATEGORICAL PIPELINE
# -----------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# -----------------------------------
# PREPROCESSOR
# -----------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# -----------------------------------
# MODEL
# -----------------------------------

model = LogisticRegression(
    max_iter=2000
)


# -----------------------------------
# COMPLETE PIPELINE
# -----------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------
# TRAIN MODEL
# -----------------------------------

print("\nTraining kidney disease model...")

pipeline.fit(
    X_train,
    y_train
)


# -----------------------------------
# EVALUATE MODEL
# -----------------------------------

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n-----------------------------------")
print("Kidney Model Accuracy:", accuracy)
print("-----------------------------------")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# -----------------------------------
# SAVE MODEL
# -----------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    pipeline,
    MODEL_PATH
)


print("\n✅ Kidney model saved successfully!")

print("Model path:")
print(MODEL_PATH)