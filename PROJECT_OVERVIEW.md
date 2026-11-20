# NeuroPredict — Prospective Bipolar Disorder Prediction Platform

## 1. Overview

**NeuroPredict** is a machine learning analytics platform built in Python and Streamlit. It predicts whether a person will receive a bipolar disorder diagnosis within a **6-month horizon**, using only **baseline predictors** measured at the start of the observation period.

The core design philosophy is **prospective, leakage-safe prediction**:

- The model does **not** rediscover the diagnostic rule (criterion circularity).
- Other diagnosis labels (Schizophrenia, Depression, Anxiety disorder, PTSD) are **excluded** as inputs to prevent label leakage.
- Model selection happens on **cross-validation/validation data**, while final performance is reported on a **held-out test set** that is never touched during tuning.

> **Important:** This project uses synthetic data for benchmark testing and educational demonstration. It does **not** represent real clinical data and is **not** intended for real-world medical deployment.

---

## 2. How the Entire System Works (End-to-End)

### Step 1 — Data Loading
The system loads `data/synthetic_future_bipolar.csv`, a synthetic cohort of **3,750 rows** and **16 columns**. Each row represents a person with 14 baseline predictors plus an ID and the target outcome (`Future_bipolar_6mo`, measured 6 months after baseline).

### Step 2 — Feature Selection & Leakage Prevention
Before any model sees the data, the following columns are removed:
- `ID` — a database key with no predictive value.
- Other diagnosis columns — these would directly reveal psychiatric status (label leakage).

The remaining 14 columns become the predictor matrix `X`, and `Future_bipolar_6mo` becomes the target `y`. All predictors are coerced to numeric, with missing values set to `NaN` for the imputer to handle.

### Step 3 — Train / Validation / Test Split
To separate model selection from final evaluation, the data is split in two stages:

1. **Outer split:** 75% train+validation / 25% held-out **test** (stratified by outcome, seed 42).
   - The test set is **completely reserved** for final evaluation. No model parameter or threshold is ever chosen using test data.

2. **Inner split:** The train+validation portion is split again into training subset and **validation** set (stratified, seed 42).
   - Models are trained on the training subset only.
   - The validation set is used exclusively for **threshold optimization** (finding the best classification probability cutoff).

This nested structure prevents information leakage from the test set into model decisions.

### Step 4 — Model Building
Three sklearn `Pipeline`s are constructed. Because preprocessing lives **inside** each pipeline, the imputer and scaler are fitted only on training data:

| Model | Pipeline Steps | Key Hyperparameters |
|-------|---------------|---------------------|
| Logistic Regression | Imputer → Scaler → Classifier | L2 penalty, `C=0.5`, `class_weight=balanced`, `max_iter=2000` |
| Random Forest | Imputer → Classifier | 100 trees, `max_depth=6`, `min_samples_leaf=10`, `class_weight=balanced` |
| Gradient Boosting | Imputer → Classifier | 50 stages, `max_depth=2`, `subsample=0.8`, `min_samples_leaf=5` |

### Step 5 — Cross-Validation (Model Selection)
A `StratifiedKFold(5, shuffle=True, seed=42)` cross-validation runs on the training+validation data:

- For each fold, each model is cloned, fitted on the fold's training portion, and evaluated on the fold's validation portion.
- Per-fold metrics (AUROC, AUPRC, Sensitivity, Specificity, Precision, F1, Accuracy, Brier) are collected.
- Results are aggregated as **mean ± standard deviation**.
- **Out-of-fold (OOF) predictions** are saved to `results/oof_predictions.csv` for use in threshold tuning.

Cross-validation tells us which model generalizes best across unseen folds, without touching the final test set.

### Step 6 — Held-Out Test Evaluation (Final Evaluation)
Each model is fitted on the full train+validation data and evaluated once on the **held-out test set**:

- **AUROC** — discrimination: ability to rank positives above negatives.
- **AUPRC** — precision-recall area, more informative than AUROC when classes are imbalanced.
- **Sensitivity / Specificity** — true positive / true negative rates.
- **Precision / F1** — positive predictive value and harmonic mean.
- **Accuracy / Brier score** — overall correctness and calibration quality.

### Step 7 — Permutation Feature Importance
For the Random Forest model, each feature column is randomly shuffled one at a time on the test set, and the drop in test AUROC is measured. Features whose shuffling causes the largest AUROC drop are the most important drivers of the model's predictions.

### Step 8 — Missing-Data Stress Test
To simulate real-world data gaps, random values in the test set are blanked out (NaN) at rates from 0% to 50%. The already-trained pipeline's imputer fills them with **training medians** (no retraining). AUROC and Brier score are measured at each rate to show robustness.

### Step 9 — Dashboard (Streamlit)
All results are loaded into a modern Streamlit dashboard with six tabs (see Section 5).

---

## 3. Machine Learning Models in Detail

### Logistic Regression
A linear classifier with L2 regularization. Because the input features are on different scales, a `StandardScaler` is placed inside the pipeline before the classifier. `class_weight=balanced` compensates for the ~15% positive class imbalance. `C=0.5` provides stronger regularization than the default, reducing overfitting.

**Strengths:** Fast, interpretable via coefficients, well-calibrated probabilities.

### Random Forest
An ensemble of 100 decision trees. Each tree is trained on a bootstrap sample with a random subset of features. `max_depth=6` and `min_samples_leaf=10` limit tree depth to prevent memorization. `class_weight=balanced` handles the imbalance.

**Strengths:** Captures non-linear interactions, robust to outliers, provides feature importance.

### Gradient Boosting
A sequential ensemble of 50 shallow trees (`max_depth=2`). Each new tree corrects errors of the previous ones. `subsample=0.8` adds stochasticity (each tree sees 80% of rows), and `min_samples_leaf=5` prevents overfitting to small leaf groups.

**Strengths:** Often strong predictive performance, handles non-linear patterns.

---

## 4. Key Concepts Explained

### Criterion Circularity
If the target label is derived from the same symptoms used as model inputs, the model is not "predicting" anything new — it is simply rediscovering the diagnostic rule. In the original `Mentalillness.csv`, manic symptoms directly defined the bipolar label, producing near-perfect but meaningless scores. NeuroPredict avoids this by using **baseline, pre-onset predictors** (demographics, history, sleep, stress, etc.) rather than current manic symptoms.

### Label Leakage
Using other diagnosis columns (e.g., `Schizophrenia`, `Depression`, `Anxiety disorder`, `PTSD`) as inputs would leak psychiatric status directly into the features. NeuroPredict excludes these columns entirely.

### Model Selection vs Final Evaluation
A common mistake is tuning hyperparameters or classification thresholds on the test set, then reporting test performance as if it were unbiased. NeuroPredict uses a **three-way split** (train / validation / test):
- **Train:** fit model parameters.
- **Validation:** choose the classification threshold.
- **Test:** report final, unbiased performance.

### Class Imbalance
The synthetic dataset has ~84.5% negative and ~15.5% positive cases. Accuracy alone is misleading; NeuroPredict emphasizes AUROC, AUPRC, Sensitivity, Specificity, and F1.

---

## 5. Dashboard — Tab-by-Tab Explanation

### Tab 1: Overview
The landing page summarizing the entire project:

- **Hero metric card:** Displays the best model's **cross-validated AUROC ± std** (not test-only), because CV is the model-selection metric. Shows model name, AUROC value, and standard deviation.
- **Dataset size:** 3,750 rows, 16 columns.
- **Positive rate:** Share of the 6-month positive class (~15.5%).
- **Missing cells:** Count of simulated missing values in the synthetic data.
- **Pipeline architecture diagram:** Visual flow — Baseline Predictors → Preprocessing → Training → Models → Evaluation, with arrows connecting each stage.
- **Project design summary:** Explains the prospective cohort setup, leakage prevention, train-only imputation, and robustness testing.

### Tab 2: Data Explorer
Explore the synthetic cohort:

- **Target distribution bar chart:** Counts of "No (6mo)" vs "Yes (6mo)" diagnoses.
- **Predictor columns:** Lists all 14 leakage-safe predictor names.
- **Dataset statistics table:** Total rows, feature count, positive rate, missing cell count.
- **Sample data:** First 10 rows of the dataframe for inspection.

### Tab 3: Training & Results
Model training and evaluation results:

- **Pipeline architecture cards:** Five stages (Baseline Predictors, Preprocessing, Training, Ensembles, Evaluation) with short descriptions.
- **Test-set metrics cards:** For each model — AUROC, AUPRC, Sensitivity, Specificity, Precision, F1, Accuracy, Brier score, with the best model highlighted by a teal left border.
- **AUROC vs AUPRC comparison chart:** Grouped bar chart comparing CV mean scores for AUROC and AUPRC across all three models, with error bars showing standard deviation. This makes it easy to see which model discriminates best and which is best at precision-recall tradeoffs.

### Tab 4: Feature Impact
Explains what drives the predictions:

- **Permutation feature importance chart:** Horizontal bar chart showing the mean drop in test AUROC when each feature is randomly shuffled. Larger drop = more important feature.
- Features shown include `Baseline_mood_instability`, `Stress_score`, `Sleep_disturbance_score`, `Age`, etc.

### Tab 5: Stress Test
Robustness under missing data:

- **AUROC degradation chart:** Line chart with area fill showing how each model's AUROC changes as simulated missingness increases from 0% to 50%. Annotations show the total AUROC drop at 50%.
- **Brier score calibration chart:** Line chart showing calibration quality (lower is better) as missingness increases.
- **Degradation summary table:** For each model — AUROC at 0% and 50%, AUROC drop, Brier at 0% and 50%, Brier rise, with progress bars for quick visual comparison.

### Tab 6: Live Prediction
Interactive prediction tool:

- **Input form:** Sliders and selectors for all 14 baseline predictors (Age, Sex, prior episodes, family history, sleep, stress, cognitive impairment, substance use, mood instability, social support, energy, social withdrawal, medication adherence).
- **Model selection:** Choose which of the three trained pipelines to use for prediction.
- **Compute Prediction button:** Runs the selected pipeline on the input row and displays the predicted probability of a future bipolar diagnosis within 6 months.
- **Result card:** Color-coded outcome — high risk (red) or low-medium risk (teal), with the exact probability.
- **Linear model weights chart:** For Logistic Regression, shows each predictor's coefficient (positive = increases risk, negative = decreases risk).

---

## 6. File Structure

```
NeuroPredict/
├── bipolar_ml.py              # Core ML pipeline (data, models, CV, metrics, stress test, plots)
├── streamlit_app.py           # Streamlit dashboard (NeuroPulse design, 6 tabs)
├── save_model.py              # Trains and saves logistic_regression_pipeline.joblib
├── predict_example.py         # Standalone prediction example script
├── generate_synthetic.py      # Synthetic dataset generator
├── PROJECT_OVERVIEW.md        # This file
├── requirements.txt           # Python dependencies
├── data/
│   ├── Mentalillness.csv      # Original cross-sectional dataset
│   └── synthetic_future_bipolar.csv  # Synthetic prospective dataset (3,750 rows)
└── results/
    ├── model_metrics.csv              # Held-out test metrics per model
    ├── cv_metrics.csv                 # 5-fold stratified CV metrics (mean ± std)
    ├── oof_predictions.csv            # Out-of-fold predictions for threshold tuning
    ├── feature_importance.csv         # Permutation importance (Random Forest)
    ├── stress_test.csv                # Missingness robustness results
    ├── logistic_regression_pipeline.joblib  # Saved fitted pipeline
    └── evaluation_plots.png           # ROC / calibration / robustness figure
```

---

## 7. How to Run

```bash
# 1. Train models and generate all results (test metrics, CV metrics, feature importance, stress test, plots)
python bipolar_ml.py --data data/synthetic_future_bipolar.csv --seed 42

# 2. Save the fitted Logistic Regression pipeline for dashboard predictions
python save_model.py

# 3. Launch the dashboard
streamlit run streamlit_app.py
```

### Command-line options for bipolar_ml.py
- `--data`: Path to input CSV (default: `data/synthetic_future_bipolar.csv`)
- `--test-size`: Fraction for held-out test set (default: 0.25)
- `--seed`: Random seed for reproducibility (default: 42)
- `--plots`: Save ROC/calibration/robustness figure to `results/`
- `--outdir`: Output directory for results (default: `results`)

---

## 8. Important Notes & Limitations

1. **Synthetic data only:** All data in `synthetic_future_bipolar.csv` is artificially generated. This is not real patient data.
2. **Not clinical deployment:** Do not use this for actual medical diagnosis or treatment decisions.
3. **No external validation:** Performance is measured on a single synthetic train/test split plus cross-validation. External validation on a separate real-world cohort is required before any clinical use.
4. **Threshold caution:** The default classification threshold of 0.50 may not be optimal for this imbalanced dataset (~15% positive rate). Use the Threshold Optimization tab to explore tradeoffs between sensitivity and specificity.
5. **Circularity avoided:** The model uses baseline predictors only — no current manic symptoms or other diagnosis labels are used as inputs.
