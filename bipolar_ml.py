"""
Machine Learning-Based Early Prediction of Bipolar Disorder
Core implementation (no dashboard) - run from the terminal:

    python bipolar_ml.py
    python bipolar_ml.py --data data/Mentalillness.csv --test-size 0.25 --seed 42 --plots

Pipeline steps (each one is a function below, in the same order):
    1. load_data                 - read CSV
    2. prepare_features_target   - remove ID + other diagnosis columns (leakage prevention)
    3. build_models              - 3 sklearn Pipelines (imputer [+ scaler] + classifier)
    4. evaluate_models           - discrimination, clinical utility, calibration metrics
    5. circularity_check         - DSM-5 manic symptom count vs label (why scores are ~perfect)
    6. permutation_feature_importance
    7. missingness_stress_test   - simulated missing-data robustness
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------
TARGET_COL = "Future_bipolar_6mo"

# Columns that must NOT be used as inputs:
#   - ID: just a database key
#   - other diagnosis labels: they directly reveal psychiatric status (label leakage)
IDENTIFIER_COLUMNS = {"ID"}
OTHER_DIAGNOSIS_COLUMNS = set()  # synthetic dataset has no leakage diagnosis columns

# The 7 manic-episode symptoms of DSM-5 Criterion B (used only for the circularity check)
MANIC_SYMPTOMS = [
    "Inflated self-esteem",
    "Decreased need for sleep",
    "More talkative than usual",
    "Racing thoughts",
    "Distractibility",
    "Increase in goal-directed activity",
    "Excessive involvement in activities with high potential for painful consequences",
]


# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
def load_data(path):
    """Read the CSV. Also tries the same folder as this script if the path is wrong."""
    if not os.path.exists(path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(path))
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"Could not find dataset at '{path}'")
    df = pd.read_csv(path)
    print(f"Loaded '{path}': {df.shape[0]} rows x {df.shape[1]} columns, "
          f"{int(df.isna().sum().sum())} missing cells")
    return df


# ----------------------------------------------------------------------
# 2. Features / target with leakage prevention
# ----------------------------------------------------------------------
def prepare_features_target(df, target_col=TARGET_COL):
    """Return X (numeric predictors), y (0/1) and the list of excluded columns."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataset")

    excluded = [c for c in df.columns
                if c in IDENTIFIER_COLUMNS or c in OTHER_DIAGNOSIS_COLUMNS]
    feature_cols = [c for c in df.columns if c != target_col and c not in excluded]

    y = df[target_col].astype(int)
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    print(f"Target '{target_col}': {int((y == 0).sum())} negative "
          f"({(y == 0).mean():.2%}), {int((y == 1).sum())} positive ({(y == 1).mean():.2%})")
    print(f"Excluded columns (leakage prevention): {excluded}")
    print(f"Number of predictor features: {len(feature_cols)}")
    return X, y, excluded


# ----------------------------------------------------------------------
# 3. Models (preprocessing lives INSIDE the pipeline -> fitted on train only)
# ----------------------------------------------------------------------
def build_models(seed):
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                penalty="l2", C=0.5, max_iter=2000, class_weight="balanced", random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=100, min_samples_leaf=10, max_depth=6,
                class_weight="balanced", random_state=seed, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GradientBoostingClassifier(
                n_estimators=50, learning_rate=0.1, max_depth=2, subsample=0.8,
                min_samples_leaf=5, random_state=seed)),
        ]),
    }


# ----------------------------------------------------------------------
# 4. Evaluation
# ----------------------------------------------------------------------
def compute_metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "AUROC": roc_auc_score(y_true, proba),
        "AUPRC": average_precision_score(y_true, proba),
        "Sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "Precision": precision_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, pred),
        "Brier": brier_score_loss(y_true, proba),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


# ----------------------------------------------------------------------
# 4b. Threshold optimization (validation-only — no test leakage)
# ----------------------------------------------------------------------
def select_optimal_threshold(y_true, proba, metric="youden"):
    """Select the classification threshold that optimizes a given metric.

    This must be run on VALIDATION / out-of-fold data only — never on the
    held-out test set — to avoid optimistic bias in the final evaluation.
    Supported metrics: 'youden' (Sensitivity + Specificity - 1),
    'balanced_accuracy', 'f1', 'precision'.
    """
    thresholds = np.linspace(0.0, 1.0, 101)
    results = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        bal_acc = (sens + spec) / 2.0
        youden = sens + spec - 1.0
        score = {"youden": youden, "balanced_accuracy": bal_acc,
                 "f1": f1, "precision": prec}[metric]
        results.append({
            "Threshold": t,
            "Sensitivity": sens,
            "Specificity": spec,
            "Precision": prec,
            "F1": f1,
            "BalancedAccuracy": bal_acc,
            "YoudenJ": youden,
            "Score": score,
        })
    df_res = pd.DataFrame(results)
    best_idx = df_res["Score"].idxmax()
    best = df_res.loc[best_idx]
    return best, df_res


def threshold_tradeoff_table(y_true, proba, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """Return a table of sensitivity/specificity at common thresholds."""
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, proba, threshold=t)
        rows.append({"Threshold": t, **m})
    return pd.DataFrame(rows)



# ----------------------------------------------------------------------
# 4c. Reliability analysis (calibration, Brier, distribution shift)
# ----------------------------------------------------------------------
def reliability_analysis(y_test, probas, X_train, X_test):
    """Assess calibration, Brier score, and distribution shift.

    Returns a dict with per-model calibration data and a shift summary
    comparing train vs test predictor distributions (KS test p-values).
    """
    from sklearn.calibration import calibration_curve

    cal_data = {}
    for name, proba in probas.items():
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y_test, proba)
        cal_data[name] = {
            "mean_pred": mean_pred,
            "frac_pos": frac_pos,
            "Brier": brier,
        }

    # Distribution shift: KS test per feature between train and test
    shift_rows = []
    for col in X_train.columns:
        from scipy.stats import ks_2samp
        try:
            stat, pval = ks_2samp(X_train[col].dropna(), X_test[col].dropna())
            shift_rows.append({
                "Feature": col,
                "KS_statistic": stat,
                "p_value": pval,
                "ShiftDetected": bool(pval < 0.05),
            })
        except Exception:
            shift_rows.append({
                "Feature": col,
                "KS_statistic": np.nan,
                "p_value": np.nan,
                "ShiftDetected": False,
            })
    shift_df = pd.DataFrame(shift_rows).sort_values("p_value")

    return cal_data, shift_df


# ----------------------------------------------------------------------
# 5. Criterion circularity check
# ----------------------------------------------------------------------
    """Fit every pipeline on TRAIN only, evaluate on the untouched TEST set."""
    rows, probas = [], {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)                      # imputer/scaler/model all fit on train
        proba = pipe.predict_proba(X_test)[:, 1]        # probability of class 1
        probas[name] = proba
        rows.append({"Model": name, **compute_metrics(y_test, proba)})
    table = pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
    return table, probas


# ----------------------------------------------------------------------
# 4b. Stratified Cross-Validation (model selection / threshold tuning data)
# ----------------------------------------------------------------------
def cross_validate_models(models, X, y, n_splits=5, seed=42):
    """Stratified K-Fold CV returning per-fold metrics and out-of-fold predictions.

    Out-of-fold predictions are used for model selection / threshold tuning,
    keeping the held-out TEST set reserved for final evaluation.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rows, oof_probas = [], {name: [] for name in models}
    oof_y_true = None

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for name, pipe in models.items():
            # Fresh clone per fold to avoid state leakage
            from sklearn.base import clone
            pipe_fold = clone(pipe)
            pipe_fold.fit(X_tr, y_tr)
            proba = pipe_fold.predict_proba(X_val)[:, 1]
            oof_probas[name].append(pd.Series(proba, index=val_idx))
            rows_metrics = compute_metrics(y_val, proba)
            rows_metrics["Model"] = name
            rows_metrics["Fold"] = fold_idx
            fold_rows.append(rows_metrics)

        if oof_y_true is None:
            oof_y_true = pd.Series(y_val.values, index=val_idx)
        else:
            oof_y_true = pd.concat([oof_y_true, pd.Series(y_val.values, index=val_idx)])

    # Concatenate out-of-fold predictions into full-sample arrays (aligned to original index)
    oof_proba_concat = {}
    for name in models:
        oof_proba_concat[name] = np.concatenate(oof_probas[name])
    oof_y_true_concat = oof_y_true.values

    # Aggregate per-fold metrics: mean +/- std
    cv_table = pd.DataFrame(fold_rows)
    agg = cv_table.groupby("Model").agg(["mean", "std"]).reset_index()
    agg.columns = ["Model"] + [f"{col[0]} ({col[1]})" if col[1] else col[0] for col in agg.columns[1:]]
    # Reorder to put mean columns before std columns
    ordered = []
    for metric in ["AUROC", "AUPRC", "Sensitivity", "Specificity", "Precision", "F1", "Accuracy", "Brier"]:
        ordered.append(f"{metric} (mean)")
        ordered.append(f"{metric} (std)")
    agg = agg[["Model"] + [m for m in ordered if m in agg.columns]]

    return agg, oof_proba_concat, oof_y_true_concat




# ----------------------------------------------------------------------
# 5. Criterion circularity check
# ----------------------------------------------------------------------
def circularity_check(df, target_col=TARGET_COL):
    """
    Count how many of the 7 DSM-5 manic symptoms each person has and compare with the label.
    If the label is almost a deterministic function of this count, the model is only
    re-discovering the diagnostic rule (circularity), not predicting anything new.
    """
    present = [c for c in MANIC_SYMPTOMS if c in df.columns]
    if not present:
        print("No DSM-5 manic symptom columns found in this dataset (prospective synthetic data). "
              "Skipping circularity check.")
        return None, None
    count = df[present].sum(axis=1)
    summary = count.groupby(df[target_col]).agg(["min", "max", "mean"])
    summary.index = [f"{target_col} = {i}" for i in summary.index]
    print("Manic-symptom count (0-7) by class:")
    print(summary.round(2).to_string())

    # simple rule: predict bipolar if >= 3 manic symptoms
    rule_pred = (count >= 3).astype(int)
    rule_acc = accuracy_score(df[target_col], rule_pred)
    print(f"\nA hand-written rule 'bipolar if >= 3 manic symptoms' reaches accuracy = {rule_acc:.4f} "
          f"with NO machine learning.")
    return summary, rule_acc


# ----------------------------------------------------------------------
# 6. Permutation feature importance
# ----------------------------------------------------------------------
def permutation_feature_importance(pipe, X_test, y_test, seed, top_n=10):
    """Drop in test AUROC when one feature column is randomly shuffled."""
    result = permutation_importance(
        pipe, X_test, y_test, n_repeats=5, random_state=seed, scoring="roc_auc", n_jobs=-1)
    imp = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean AUROC drop": result.importances_mean,
        "Std": result.importances_std,
    }).sort_values("Mean AUROC drop", ascending=False).head(top_n).reset_index(drop=True)
    return imp


# ----------------------------------------------------------------------
# 7. Simulated missing-data stress test
# ----------------------------------------------------------------------
def missingness_stress_test(models, X_test, y_test, seed, rates=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """
    Randomly blank out a fraction of TEST values (NaN) and let the already-trained
    pipeline's imputer fill them with training medians. No retraining.
    """
    rows = []
    for rate in rates:
        rng = np.random.default_rng(seed)               # same mask pattern for every model
        mask = rng.random(X_test.shape) < rate
        X_missing = X_test.mask(mask)
        for name, pipe in models.items():
            proba = pipe.predict_proba(X_missing)[:, 1]
            rows.append({
                "Missing %": int(rate * 100),
                "Model": name,
                "AUROC": roc_auc_score(y_test, proba),
                "Brier": brier_score_loss(y_test, proba),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Optional plots
# ----------------------------------------------------------------------
def save_plots(y_test, probas, stress_df, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    from sklearn.calibration import calibration_curve

    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax[0].plot(fpr, tpr, label=name)
        frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10)
        ax[1].plot(mean_pred, frac_pos, marker="o", label=name)
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].plot([0, 1], [0, 1], "k--"); ax[1].set(title="Calibration curve", xlabel="Mean predicted probability", ylabel="Observed fraction positive")

    for name, sub in stress_df.groupby("Model"):
        ax[2].plot(sub["Missing %"], sub["AUROC"], marker="o", label=name)
    ax[2].set(title="AUROC vs simulated missingness", xlabel="Missing values (%)", ylabel="AUROC")

    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, "evaluation_plots.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plots -> {path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Bipolar disorder ML baseline (no dashboard)")
    parser.add_argument("--data", default=os.path.join("data", "synthetic_future_bipolar.csv"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plots", action="store_true", help="save ROC/calibration/robustness figure")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    def banner(t):
        print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

    banner("1-2. DATA LOADING & LEAKAGE-SAFE FEATURE SELECTION")
    df = load_data(args.data)
    X, y, _ = prepare_features_target(df)

    banner("3. STRATIFIED TRAIN+VAL / TEST SPLIT (test held out for final eval)")
    # Stage boundary: split off the held-out TEST set first. Everything before
    # this point (training + validation) is used for model selection only.
    X_trv, X_test, y_trv, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    print(f"Train+Val: {len(X_trv)} rows | Test: {len(X_test)} rows "
          f"| positive rate trv={y_trv.mean():.3f}, test={y_test.mean():.3f}")

    banner("4. TRAINING: fit models on Train+Val")
    models = build_models(args.seed)
    for name, pipe in models.items():
        pipe.fit(X_trv, y_trv)  # imputer/scaler/classifier all fit on Train+Val only

    banner("5. VALIDATION: stratified CV on Train+Val + threshold selection")
    # CV runs ONLY on Train+Val (never touches the held-out test set)
    cv_table, oof_probas, oof_y_true = cross_validate_models(
        build_models(args.seed), X_trv, y_trv, n_splits=5, seed=args.seed)
    print("\nSTRATIFIED CROSS-VALIDATION (5 folds, mean +/- std):")
    print(cv_table.round(4).to_string(index=False))

    # Threshold optimization on validation (OOF) data — selects a threshold per model
    print("\nTHRESHOLD OPTIMIZATION (validation/OOF data, metric=Youden's J):")
    selected_thresholds = {}
    for name in models:
        best, _ = select_optimal_threshold(oof_y_true, oof_probas[name], metric="youden")
        selected_thresholds[name] = float(best["Threshold"])
        print(f"  {name}: threshold={best['Threshold']:.2f} "
              f"(Sens={best['Sensitivity']:.3f}, Spec={best['Specificity']:.3f}, "
              f"F1={best['F1']:.3f})")

    banner("6. TESTING: final evaluation on held-out TEST set (untouched)")
    # Final evaluation uses the models trained on Train+Val, scored on the
    # held-out test set, with thresholds selected from validation data.
    test_rows = []
    for name, pipe in models.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        thr = selected_thresholds[name]
        m = compute_metrics(y_test, proba, threshold=thr)
        m["Model"] = name
        m["Threshold"] = thr
        test_rows.append(m)
    test_table = pd.DataFrame(test_rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
    print(test_table.round(4).to_string(index=False))

    banner("7. RELIABILITY ANALYSIS (calibration, Brier, distribution shift)")
    cal_data, shift_df = reliability_analysis(y_test, {n: p.predict_proba(X_test)[:, 1]
                                                       for n, p in models.items()},
                                              X_trv, X_test)
    print("\nCalibration / Brier (test set):")
    for name, cd in cal_data.items():
        print(f"  {name}: Brier={cd['Brier']:.4f}")
    print("\nDistribution shift (train vs test, KS test):")
    print(shift_df.round(4).to_string(index=False))



    banner("5. CRITERION CIRCULARITY CHECK")
    circularity_check(df)

    banner("6. PERMUTATION FEATURE IMPORTANCE (Random Forest, test AUROC drop)")
    imp = permutation_feature_importance(models["Random Forest"], X_test, y_test, args.seed)
    print(imp.round(4).to_string(index=False))

    banner("7. SIMULATED MISSING-DATA STRESS TEST")
    stress = missingness_stress_test(models, X_test, y_test, args.seed)
    print(stress.pivot(index="Missing %", columns="Model", values="AUROC").round(4).to_string())
    print("\n(Brier score)")
    print(stress.pivot(index="Missing %", columns="Model", values="Brier").round(4).to_string())

    os.makedirs(args.outdir, exist_ok=True)
    test_table.to_csv(os.path.join(args.outdir, "model_metrics.csv"), index=False)
    cv_table.to_csv(os.path.join(args.outdir, "cv_metrics.csv"), index=False)
    imp.to_csv(os.path.join(args.outdir, "feature_importance.csv"), index=False)
    stress.to_csv(os.path.join(args.outdir, "stress_test.csv"), index=False)
    shift_df.to_csv(os.path.join(args.outdir, "distribution_shift.csv"), index=False)

    # Save selected thresholds + reliability summary
    thr_df = pd.DataFrame([
        {"Model": n, "SelectedThreshold": selected_thresholds[n]} for n in models
    ])
    thr_df.to_csv(os.path.join(args.outdir, "selected_thresholds.csv"), index=False)
    rel_rows = []
    for name, cd in cal_data.items():
        rel_rows.append({"Model": name, "Brier": cd["Brier"]})
    rel_df = pd.DataFrame(rel_rows)
    rel_df.to_csv(os.path.join(args.outdir, "reliability_brier.csv"), index=False)

    # Save out-of-fold predictions for threshold optimization (dashboard use)
    oof_df = pd.DataFrame(oof_probas)
    oof_df["y_true"] = oof_y_true
    oof_df.to_csv(os.path.join(args.outdir, "oof_predictions.csv"), index=False)
    print(f"\nSaved CSV results -> '{args.outdir}/' "
          f"(incl. model_metrics.csv, cv_metrics.csv, selected_thresholds.csv, "
          f"reliability_brier.csv, distribution_shift.csv, oof_predictions.csv)")
    if args.plots:
        test_probas = {n: p.predict_proba(X_test)[:, 1] for n, p in models.items()}
        save_plots(y_test, test_probas, stress, args.outdir)

    banner("IMPORTANT NOTE")
    print("This run uses SYNTHETIC prospective data (Future_bipolar_6mo at 6 months).\n"
          "Three-stage pipeline: Training (fit on Train+Val) -> Validation (CV + "
          "threshold selection on OOF data) -> Testing (held-out test set, untouched).\n"
          "All three models are retained; none is auto-dropped based on test AUROC.\n"
          "Scores reflect prediction from baseline predictors, NOT current-state diagnosis.\n"
          "Not external validation; intended for pipeline/benchmark testing only.")


if __name__ == "__main__":
    main()
