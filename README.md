# GlycoLens — Diabetes Prediction System

A complete educational machine-learning application for exploring classification on the Pima Indians Diabetes dataset. It trains three classifiers, evaluates them on a held-out stratified test split, persists the complete preprocessing/model pipelines, and provides a polished Streamlit dashboard for individual and batch inference.

> **Educational disclaimer:** This application is not a medical diagnostic tool. It is not validated for clinical use, does not estimate an individual's clinical risk, and must not be used to make health decisions. Seek advice from a qualified healthcare professional.

## Dataset

The classic Pima Indians Diabetes Database is distributed as a headerless CSV mirror at [mikeizbicki/datasets](https://github.com/mikeizbicki/datasets/blob/master/csv/uci/pima-indians-diabetes.csv) (raw file: `https://raw.githubusercontent.com/mikeizbicki/datasets/master/csv/uci/pima-indians-diabetes.csv`). The included `data/diabetes.csv` has 768 records, eight predictors, and a binary `Outcome` label. The dataset has important population and historical limitations and is not representative of all patients.

Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age. Target: Outcome (0/1).

## Methodology

- Data validated against the expected 768×9 schema; class counts and positive rate are recorded in evaluation metadata.
- In Glucose, BloodPressure, SkinThickness, Insulin, and BMI, zero is treated as an invalid/missing measurement. These values are imputed using a median imputer fit on training data, with missingness indicators; features are standardized in the same pipeline. Other zero values (for example Pregnancies) remain valid. Blank CSV cells in these five fields are also treated as missing; blanks in other features are rejected.
- A single 80/20 stratified split with seed 42 is used. Preprocessing is fit only as part of each training pipeline. This is a transparent demonstration, not extensive hyperparameter search; models are not tuned against the test set.
- Classifiers: class-weighted Logistic Regression, Random Forest (400 trees, minimum leaf size 3), and RBF Support Vector Machine. Each full pipeline is saved as a joblib artifact.
- Reported metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC curve data. The chosen reference model is the model with the highest test-split F1. This simple selection rule is educational and does not account for clinical costs.
- Feature interpretation shows Random Forest impurity-based importance only. Importance is model-specific, not causal, not directional, and not a medical explanation.

## Features

- Overview dashboard with actual dataset counts, class distribution, held-out evaluation comparison, and current F1-leading model.
- Dark appearance is the first-visit default; the selected appearance is remembered in the URL (`?theme=light` or `?theme=dark`) across reloads and can be bookmarked/shared.
- Individual inference with input validation, model selection, optional comparison of all three saved models, class-1 probability caveat, and SQLite history recording.
- Model comparison metrics, interactive metric chart, confusion matrices, and ROC curves.
- CSV batch inference (up to 10,000 records) and downloadable predictions. Required feature columns are validated; extra columns such as Outcome are ignored.
- Random Forest feature-importance chart and table.
- Local SQLite prediction history, with explicit confirmation before clearing.
- Custom responsive CSS; interactive Plotly visualizations.

## Project layout

```text
app.py                     Streamlit interface
requirements.txt           Python dependencies
README.md                  This documentation
assets/styles.css          App styling tokens
assets/glycolens-logo.svg    Custom GlycoLens app logo
.streamlit/config.toml      Light/dark defaults for Streamlit-native widgets
data/diabetes.csv           Included dataset
models/*.joblib             Persisted trained pipelines (after training)
models/evaluation.json      Actual held-out test outputs and metadata
src/preprocessing.py        Dataset schema and preprocessing pipeline
src/train.py                Reproducible model training/evaluation
src/predict.py              Loading, validation, inference helpers
src/history.py              SQLite history helpers
tests/test_system.py        Automated integration tests
```

## Setup and launch

Python 3.10–3.13 recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.train
python -m unittest discover -s tests -v
streamlit run app.py
```

Training can be rerun with `python -m src.train`; it overwrites model artifacts and evaluation JSON. The app loads saved artifacts and does not retrain at startup. Keep the `src` folder as a Python package and start from the project root.

## Evaluation results

The actual metrics and per-model confusion matrices for the included training run are in `models/evaluation.json` and are printed by the training command. Results depend on the explicit split and package versions; avoid comparing them as clinical benchmarks. See the evaluation JSON for the precise generated values.

## Limitations

This is a small historical dataset with a restricted source population and known measurement/missingness issues. A single test split is noisy; no external validation, calibration assessment, prospective evaluation, fairness audit, or clinical review has been performed. Predicted probabilities are not calibrated medical risk estimates. The application logs measurements locally in `data/history.sqlite3`; protect or delete that file if using any real data, and do not enter identifiable or real patient information.
