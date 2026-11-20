"""Dataset loading and leakage-safe preprocessing helpers."""
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"
COLUMNS = FEATURES + [TARGET]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    if df.shape != (768, 9) or not set(df[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("Dataset did not match the expected Pima schema (768 rows, 9 columns).")
    for col in ZERO_AS_MISSING:
        df.loc[df[col] == 0, col] = float("nan")
    return df


def make_preprocessor() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())])
