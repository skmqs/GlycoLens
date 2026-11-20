"""Reproducible train/test evaluation and model persistence."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from .preprocessing import FEATURES, TARGET, load_data, make_preprocessor

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def train(data_path=ROOT / "data" / "diabetes.csv", model_dir=ROOT / "models"):
    df = load_data(data_path)
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, stratify=y, random_state=SEED)
    specs = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Support Vector Machine": SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
    }
    model_dir = Path(model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    results = {"metadata": {"seed": SEED, "test_size": .2, "rows": len(df), "class_counts": y.value_counts().sort_index().to_dict(), "train_rows": len(X_train), "test_rows": len(X_test), "positive_rate": float(y.mean())}, "models": {}}
    for name, estimator in specs.items():
        pipe = Pipeline([("preprocessor", make_preprocessor()), ("classifier", clone(estimator))])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        results["models"][name] = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0), "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0), "roc_auc": roc_auc_score(y_test, prob), "confusion_matrix": confusion_matrix(y_test, pred).tolist(), "roc": {"fpr": roc_curve(y_test, prob)[0].tolist(), "tpr": roc_curve(y_test, prob)[1].tolist()}, "test_predictions": pred.astype(int).tolist()}
        joblib.dump(pipe, model_dir / (name.lower().replace(" ", "_") + ".joblib"))
    # Importance only from the forest, clearly labeled as model-specific.
    forest = joblib.load(model_dir / "random_forest.joblib")
    results["feature_importance"] = dict(zip(FEATURES, forest.named_steps["classifier"].feature_importances_.tolist()))
    results["test_indices"] = [int(i) for i in X_test.index]
    results["test_actual"] = y_test.astype(int).tolist()
    (model_dir / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results

if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
