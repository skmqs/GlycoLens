"""Load saved pipelines and validate prediction inputs."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .preprocessing import FEATURES, ZERO_AS_MISSING
ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = {"Logistic Regression": "logistic_regression.joblib", "Random Forest": "random_forest.joblib", "Support Vector Machine": "support_vector_machine.joblib"}

def load_models(model_dir=ROOT / "models"):
    return {name: joblib.load(Path(model_dir) / file) for name, file in MODEL_FILES.items()}

def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing: raise ValueError("Missing required columns: " + ", ".join(missing))
    original = frame[FEATURES].copy()
    out = original.copy()
    for c in FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if (original[c].notna() & out[c].isna()).any():
            raise ValueError(f"{c} contains a non-numeric value.")
        if c not in ZERO_AS_MISSING and out[c].isna().any():
            raise ValueError(f"{c} cannot be blank.")
    if not np.isfinite(out.fillna(0).to_numpy(dtype=float)).all(): raise ValueError("Input must contain finite numeric values.")
    bounds = {"Pregnancies": (0, 30), "Glucose": (0, 300), "BloodPressure": (0, 200), "SkinThickness": (0, 100), "Insulin": (0, 1000), "BMI": (0, 90), "DiabetesPedigreeFunction": (0, 3), "Age": (1, 120)}
    for c, (lo, hi) in bounds.items():
        if ((out[c] < lo) | (out[c] > hi)).any(): raise ValueError(f"{c} must be between {lo} and {hi}.")
    for c in ZERO_AS_MISSING: out.loc[out[c] == 0, c] = np.nan
    return out

def predict_frame(frame, models=None):
    models = models or load_models()
    X = validate_frame(frame)
    result = frame.copy()
    for name, model in models.items():
        result[name + " prediction"] = model.predict(X).astype(int)
        result[name + " probability"] = model.predict_proba(X)[:, 1]
    return result
