"""
train_model.py
---------------
Trains RandomForest / GradientBoosting regressors to predict:
  1. Volumetric Expansion (%) from in-situ spinel formation
  2. Slag Corrosion Index

from composition (Al2O3, MgO, SiO2) and process parameters (sinter temp,
hold time). Reports cross-validated performance and feature importances,
and saves trained models + plots to outputs/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

DATA_PATH = "/home/claude/topic1_spinel_refractory/data/spinel_data.csv"
OUT_DIR = "/home/claude/topic1_spinel_refractory/outputs"

FEATURES = ["Al2O3_pct", "MgO_pct", "SiO2_pct", "Sinter_Temp_C", "Hold_Time_hr"]
TARGETS = ["Volumetric_Expansion_pct", "Slag_Corrosion_Index"]


def load_data():
    return pd.read_csv(DATA_PATH)


def train_and_evaluate(df, target, model_type="rf"):
    X = df[FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=2,
            random_state=42
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            random_state=42
        )

    # 5-fold CV on the training set (small-data best practice: don't trust a
    # single train/test split; report the distribution of CV scores).
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")
    cv_mae = -cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    print(f"
=== Target: {target} | Model: {model_type.upper()} ===")
    print(f"CV R^2:  {cv_r2.mean():.3f} +/- {cv_r2.std():.3f}")
    print(f"CV MAE:  {cv_mae.mean():.4f} +/- {cv_mae.std():.4f}")
    print(f"Held-out test R^2: {test_r2:.3f} | MAE: {test_mae:.4f}")

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("Feature importances:")
    print(importances.round(3).to_string())

    joblib.dump(model, f"{OUT_DIR}/model_{target}_{model_type}.joblib")

    return {
        "model": model, "cv_r2": cv_r2, "cv_mae": cv_mae,
        "test_r2": test_r2, "test_mae": test_mae,
        "importances": importances, "X_test": X_test, "y_test": y_test, "y_pred": y_pred
    }


def plot_results(results_by_target):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, (target, res) in enumerate(results_by_target.items()):
        # Predicted vs actual
        ax = axes[0, i]
        ax.scatter(res["y_test"], res["y_pred"], alpha=0.7, edgecolor="k")
        lims = [min(res["y_test"].min(), res["y_pred"].min()),
                max(res["y_test"].max(), res["y_pred"].max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="Ideal")
        ax.set_xlabel(f"Actual {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"{target}
Test R^2 = {res['test_r2']:.3f}")
        ax.legend()

        # Feature importance
        ax2 = axes[1, i]
        res["importances"].sort_values().plot(kind="barh", ax=ax2, color="steelblue")
        ax2.set_title(f"Feature Importance: {target}")
        ax2.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/model_performance.png", dpi=150)
    print(f"
Saved plot to {OUT_DIR}/model_performance.png")


if __name__ == "__main__":
    df = load_data()
    results = {}
    for target in TARGETS:
        results[target] = train_and_evaluate(df, target, model_type="rf")
    plot_results(results)
}