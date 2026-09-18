"""
Train RandomForest regressors for porosity, MOR, and dissolution rate and
compare held-out predictive performance with permutation importance.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import joblib

DATA_PATH = "/home/claude/topic2_ceramic_core/data/leaching_data.csv"
OUT_DIR = "/home/claude/topic2_ceramic_core/outputs"
FEATURES = ["Grain_Size_um", "Sinter_Temp_C", "Hold_Time_hr", "KOH_Conc_M", "Bath_Temp_C"]
TARGETS = ["Porosity_pct", "MOR_MPa", "Dissolution_Rate_mg_cm2_min"]


def train_target(df, target):
    X, y = df[FEATURES], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    perm = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
    importance = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    print(f"{target}: CV R2={cv_r2.mean():.3f} +/- {cv_r2.std():.3f}; "
          f"test R2={r2_score(y_test, pred):.3f}; MAE={mean_absolute_error(y_test, pred):.3f}")
    joblib.dump(model, f"{OUT_DIR}/model_{target}.joblib")
    return {"model": model, "test_r2": r2_score(y_test, pred), "test_mae": mean_absolute_error(y_test, pred),
            "importance": importance, "y_test": y_test, "pred": pred}


def plot_all(results):
    fig, axes = plt.subplots(len(TARGETS), 2, figsize=(12, 4 * len(TARGETS)))
    for i, target in enumerate(TARGETS):
        res = results[target]
        ax = axes[i, 0]
        ax.scatter(res["y_test"], res["pred"], alpha=0.7)
        lo = min(res["y_test"].min(), res["pred"].min())
        hi = max(res["y_test"].max(), res["pred"].max())
        ax.plot([lo, hi], [lo, hi], "r--")
        ax.set(xlabel=f"Actual {target}", ylabel=f"Predicted {target}", title=f"{target} | Test R2={res['test_r2']:.3f}")
        res["importance"].sort_values().plot(kind="barh", ax=axes[i, 1])
        axes[i, 1].set(title=f"Permutation Importance: {target}", xlabel="Mean importance")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/leaching_model_performance.png", dpi=150)


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    results = {target: train_target(df, target) for target in TARGETS}
    plot_all(results)
    print("Dominant dissolution-rate driver:", results["Dissolution_Rate_mg_cm2_min"]["importance"].idxmax())
