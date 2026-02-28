from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _build_boosting_grid(cv):
    """Try LightGBM first; if unavailable (e.g., missing libomp on macOS), fallback to sklearn HGB."""
    try:
        import lightgbm as lgb

        grid = GridSearchCV(
            lgb.LGBMClassifier(objective="binary", random_state=RANDOM_STATE),
            {"n_estimators": [50, 100, 200], "max_depth": [3, 4, 5, 6], "learning_rate": [0.01, 0.05, 0.1]},
            scoring="f1",
            cv=cv,
            n_jobs=-1,
        )
        return "LightGBM", grid, None
    except Exception as exc:  # noqa: BLE001
        grid = GridSearchCV(
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 8],
                "max_iter": [100, 200, 300],
            },
            scoring="f1",
            cv=cv,
            n_jobs=-1,
        )
        return "Gradient Boosting (fallback)", grid, str(exc)


def _plot_shap(model, X_test, summary_path: Path, bar_path: Path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(vals, X_test, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(vals, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150)
    plt.close()


def main(data_path: Path):
    root = Path(__file__).parent
    artifacts = root / "artifacts"
    plots = artifacts / "plots"
    models_dir = artifacts / "models"
    plots.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path, usecols=lambda c: c != "Unnamed: 0")

    fig = plt.figure(figsize=(6, 4))
    sns.countplot(x="DEATH", data=data)
    plt.title("Target Distribution")
    plt.tight_layout()
    fig.savefig(plots / "target_distribution.png", dpi=150)
    plt.close(fig)

    death_1 = data[data["DEATH"] == 1].sample(n=min(5000, int((data["DEATH"] == 1).sum())), random_state=RANDOM_STATE)
    death_0 = data[data["DEATH"] == 0].sample(n=min(5000, int((data["DEATH"] == 0).sum())), random_state=RANDOM_STATE)
    df = pd.concat([death_1, death_0], ignore_index=True)

    fig = plt.figure(figsize=(7, 5))
    sns.boxplot(x="DEATH", y="AGE", data=df)
    plt.tight_layout()
    fig.savefig(plots / "age_by_outcome_boxplot.png", dpi=150)
    plt.close(fig)

    df_viz = df.copy()
    df_viz["AGE_GROUP"] = pd.cut(
        df_viz["AGE"], bins=[0, 29, 39, 49, 59, 69, 79, 120], labels=["<30", "30–39", "40–49", "50–59", "60–69", "70–79", "80+"]
    )
    age_mortality = df_viz.groupby("AGE_GROUP", observed=False)["DEATH"].mean().reset_index()
    fig = plt.figure(figsize=(8, 4))
    sns.barplot(x="AGE_GROUP", y="DEATH", data=age_mortality)
    plt.ylim(0, 1)
    plt.tight_layout()
    fig.savefig(plots / "mortality_by_age_group.png", dpi=150)
    plt.close(fig)

    comorbidities = ["DIABETES", "HYPERTENSION", "OBESITY", "RENAL_CHRONIC", "CARDIOVASCULAR"]
    rows = []
    for c in comorbidities:
        rates = df.groupby(c)["DEATH"].mean()
        rows += [
            {"Condition": c, "Status": "No", "MortalityRate": float(rates.loc[0])},
            {"Condition": c, "Status": "Yes", "MortalityRate": float(rates.loc[1])},
        ]
    mort_df = pd.DataFrame(rows)
    fig = plt.figure(figsize=(10, 4))
    sns.barplot(x="Condition", y="MortalityRate", hue="Status", data=mort_df)
    plt.ylim(0, 1)
    plt.tight_layout()
    fig.savefig(plots / "mortality_by_comorbidity.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.tight_layout()
    fig.savefig(plots / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    X = df.drop(columns="DEATH")
    y = df["DEATH"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"min_samples_leaf": [40, 50, 100, 200], "max_depth": [4, 5, 6, 9]},
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [25, 50, 100, 200], "max_depth": [3, 5, 8, 10]},
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    boosting_name, boosting_grid, boosting_import_error = _build_boosting_grid(cv)

    dt_grid.fit(X_train, y_train)
    rf_grid.fit(X_train, y_train)
    boosting_grid.fit(X_train, y_train)

    models = {
        "Decision Tree": dt_grid.best_estimator_,
        "Random Forest": rf_grid.best_estimator_,
        boosting_name: boosting_grid.best_estimator_,
    }

    metrics = {}
    for name, model in models.items():
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred)),
            "recall": float(recall_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "auc": float(roc_auc_score(y_test, prob)),
        }

    best_params = {
        "Decision Tree": dt_grid.best_params_,
        "Random Forest": rf_grid.best_params_,
        boosting_name: boosting_grid.best_params_,
    }

    defaults = {c: float(X_train[c].mean()) for c in X_train.columns}
    ranges = {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max())} for c in X_train.columns}

    shap_model = models.get("LightGBM", models["Random Forest"])
    _plot_shap(shap_model, X_test, plots / "shap_summary.png", plots / "shap_bar.png")

    joblib.dump(
        {
            "models": models,
            "feature_names": list(X_train.columns),
            "X_train": X_train,
            "X_test": X_test,
            "y_test": y_test,
        },
        models_dir / "model_bundle.joblib",
    )

    save_json(artifacts / "model_metrics.json", metrics)
    save_json(artifacts / "best_params.json", best_params)
    save_json(artifacts / "feature_defaults.json", defaults)
    save_json(artifacts / "feature_ranges.json", ranges)

    run_notes = {
        "boosting_model_used": boosting_name,
        "lightgbm_import_error": boosting_import_error,
    }
    save_json(artifacts / "run_notes.json", run_notes)

    print("Artifacts generated in ./artifacts")
    if boosting_import_error:
        print("LightGBM unavailable; used sklearn fallback model.")
        print(f"Reason: {boosting_import_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("covid.csv"))
    args = parser.parse_args()
    main(args.data)
