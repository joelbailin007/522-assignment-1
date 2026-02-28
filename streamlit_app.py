from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import RocCurveDisplay

st.set_page_config(page_title="COVID-19 Mortality Risk App", layout="wide")

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
MODELS_DIR = ART / "models"
PLOTS_DIR = ART / "plots"


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_assets():
    assets = {
        "model_bundle": None,
        "metrics": _load_json(ART / "model_metrics.json"),
        "best_params": _load_json(ART / "best_params.json"),
        "feature_defaults": _load_json(ART / "feature_defaults.json"),
        "feature_ranges": _load_json(ART / "feature_ranges.json"),
        "run_notes": _load_json(ART / "run_notes.json"),
    }

    model_path = MODELS_DIR / "model_bundle.joblib"
    if model_path.exists():
        assets["model_bundle"] = joblib.load(model_path)
    return assets


assets = load_assets()
bundle = assets["model_bundle"]

st.title("COVID-19 Mortality Risk — Streamlit Deployment")
st.caption("All models are loaded from saved artifacts; no training is performed in the app runtime.")

if bundle is None:
    st.error(
        "Missing model artifacts. Run `python build_artifacts.py --data covid.csv` first to create pretrained models and plots."
    )


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Summary",
        "Descriptive Analytics",
        "Model Performance",
        "Explainability & Interactive Prediction",
    ]
)

with tab1:
    st.header("Executive Summary")
    st.markdown(
        """
        This project predicts **COVID-19 mortality risk** (`DEATH`) from patient demographics and clinical indicators.

        **Why this matters:** mortality risk stratification supports triage, resource allocation, and earlier intervention for high-risk patients.

        **Approach:** compare Decision Tree, Random Forest, and a boosted model (LightGBM when available, with a robust fallback) using cross-validation, test-set metrics, ROC curves, and SHAP explainability.

        This tab is intentionally non-technical and focused on stakeholder-level findings.
        """
    )

    metrics = assets.get("metrics") or {}
    if metrics:
        cols = st.columns(min(3, len(metrics)))
        for i, (name, m) in enumerate(metrics.items()):
            with cols[i % len(cols)]:
                st.metric(f"{name} — F1", f"{m.get('f1', float('nan')):.3f}")
                st.metric(f"{name} — AUC", f"{m.get('auc', float('nan')):.3f}")

with tab2:
    st.header("Descriptive Analytics")
    st.write("Key Part 1 visualizations with brief commentary.")

    plot_files = [
        ("target_distribution.png", "Target distribution: class balance for mortality outcome."),
        ("age_by_outcome_boxplot.png", "Age is visibly higher for deceased patients."),
        ("mortality_by_age_group.png", "Mortality rate increases strongly with age group."),
        ("mortality_by_comorbidity.png", "Comorbidities elevate mortality risk."),
        ("correlation_heatmap.png", "Correlation heatmap across features and target."),
    ]
    for fn, cap in plot_files:
        p = PLOTS_DIR / fn
        if p.exists():
            st.image(str(p), caption=cap, use_container_width=True)
        else:
            st.info(f"Plot not found: {fn}")

with tab3:
    st.header("Model Performance")

    metrics = assets.get("metrics")
    params = assets.get("best_params")

    if metrics:
        df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
        st.subheader("Model comparison table")
        st.dataframe(df, use_container_width=True)

        st.subheader("F1 comparison")
        st.bar_chart(df.set_index("model")[["f1"]])

    if params:
        st.subheader("Best hyperparameters")
        st.json(params)

    run_notes = assets.get("run_notes") or {}
    if run_notes.get("lightgbm_import_error"):
        st.warning("LightGBM could not be loaded in this environment; fallback boosting model was used for artifact generation.")

    if bundle is not None and "X_test" in bundle and "y_test" in bundle:
        st.subheader("ROC curves")
        X_test = bundle["X_test"]
        y_test = bundle["y_test"]
        fig, ax = plt.subplots(figsize=(7, 5))
        for model_name, model in bundle["models"].items():
            if hasattr(model, "predict_proba"):
                RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=model_name)
        st.pyplot(fig)

with tab4:
    st.header("Explainability & Interactive Prediction")

    if bundle is None:
        st.stop()

    models: Dict[str, object] = bundle["models"]
    feature_names: List[str] = bundle["feature_names"]
    X_train = bundle["X_train"]

    st.subheader("SHAP summary")
    shap_summary = PLOTS_DIR / "shap_summary.png"
    shap_bar = PLOTS_DIR / "shap_bar.png"
    if shap_summary.exists():
        st.image(str(shap_summary), caption="SHAP summary plot", use_container_width=True)
    if shap_bar.exists():
        st.image(str(shap_bar), caption="SHAP feature importance bar plot", use_container_width=True)

    st.subheader("Interactive prediction")
    model_name = st.selectbox("Select model", list(models.keys()))
    model = models[model_name]

    defaults = assets.get("feature_defaults") or {c: float(np.mean(X_train[c])) for c in feature_names}
    ranges = assets.get("feature_ranges") or {
        c: {"min": float(np.min(X_train[c])), "max": float(np.max(X_train[c]))} for c in feature_names
    }

    selected_features = st.multiselect(
        "Choose features to manually set", feature_names, default=feature_names[: min(8, len(feature_names))]
    )
    input_row = {}

    for col in feature_names:
        if col in selected_features:
            r = ranges.get(col, {"min": 0.0, "max": 1.0})
            mn, mx = float(r.get("min", 0.0)), float(r.get("max", 1.0))
            if abs(mx - mn) < 1e-12:
                mx = mn + 1.0
            input_row[col] = st.slider(col, min_value=mn, max_value=mx, value=float(defaults.get(col, mn)))
        else:
            input_row[col] = float(defaults.get(col, 0.0))

    X_input = pd.DataFrame([input_row], columns=feature_names)
    pred = int(model.predict(X_input)[0])

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X_input)[0, 1])
        st.success(f"Predicted class: {pred} (death probability = {proba:.3f})")
    else:
        st.success(f"Predicted class: {pred}")

    st.subheader("SHAP waterfall for this custom input")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)
        if isinstance(shap_values, list):
            vals = shap_values[1]
        else:
            vals = shap_values

        exp = shap.Explanation(
            values=np.array(vals[0]),
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
            data=X_input.iloc[0].values,
            feature_names=feature_names,
        )
        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(exp, max_display=12, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not generate SHAP waterfall for selected model: {e}")
