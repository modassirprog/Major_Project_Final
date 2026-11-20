# Code Explanation & Viva Preparation Guide

**File explained:** `bipolar_ml.py`
**Run it:** `pip install -r requirements.txt` then `python bipolar_ml.py --plots`

---

## 0. The 30-second summary (say this first)

> "My project builds a machine learning baseline that classifies bipolar disorder from 53 symptom columns. I train three models (Logistic Regression, Random Forest, Gradient Boosting) with a leakage-safe pipeline, evaluate them on discrimination, clinical utility and calibration, and test robustness to missing data. The models score almost perfectly, and I show that this happens because the label is defined by the same symptoms used as input (criterion circularity). So this is a current-state classification baseline, not early prediction, until I get longitudinal and multimodal data."

---

## 1. Data facts (from the actual CSV)

| Item | Value |
|---|---|
| Rows / columns | 3,753 x 59 |
| Missing cells | 0 |
| Target | `Bipolar disorder` (1 = 2,909 = 77.51%, 0 = 844 = 22.49%) |
| Excluded columns | `ID`, `Schizophrenia`, `Depression`, `Anxiety disorder`, `PTSD` |
| Predictors used | 53 binary symptom columns |
| Split | 75% train (2,814) / 25% test (939), stratified, seed 42 |

---

## 2. Code walkthrough, function by function

### Imports and constants
- `TARGET_COL`: the column we predict.
- `IDENTIFIER_COLUMNS`, `OTHER_DIAGNOSIS_COLUMNS`: columns we deliberately **do not** feed to the model.
- `MANIC_SYMPTOMS`: the 7 DSM-5 Criterion B manic symptoms. Used **only** in the circularity check, not for training.

### Step 1 `load_data(path)`
Reads the CSV with pandas and prints shape and the missing-cell count. It also checks the script's own folder if the path is wrong, so it will not crash in the demo.

### Step 2 `prepare_features_target(df)`
- Builds the list of excluded columns, then `feature_cols` = everything except target and excluded.
- `y` is the target as 0/1 integers.
- `X` is the predictors converted to numbers (`pd.to_numeric(errors="coerce")`, so bad entries become NaN and the imputer handles them).
- **Why exclude the other diagnoses?** If `Schizophrenia`, `Depression`, etc. were inputs, the model could use other labels as shortcuts. This is **label leakage**. `ID` carries no clinical meaning.

### Step 3 `train_test_split(..., stratify=y)` (inside `main`)
- Holds out 25% of the data that models never see during training.
- `stratify=y` keeps the 77.5% / 22.5% ratio in both train and test, so the test set is a fair mini-copy of the data.
- `random_state=42` makes the result reproducible.

### Step 4 `build_models(seed)`
Returns three scikit-learn `Pipeline` objects. A pipeline chains the steps so they run in order.

| Model | Pipeline | Why |
|---|---|---|
| **Logistic Regression** | median imputer, StandardScaler, LogReg (L2, `class_weight="balanced"`, `max_iter=2000`) | Interpretable linear baseline. Scaling is needed because it is a gradient/distance-sensitive linear model. |
| **Random Forest** | median imputer, 200 trees, `min_samples_leaf=2`, `class_weight="balanced"` | Non-linear; many decorrelated trees vote (bagging), which reduces variance. |
| **Gradient Boosting** | median imputer, 100 stages, `learning_rate=0.1`, `max_depth=3` | Non-linear; trees are built one after another, each correcting the previous errors. |

**Key point: why a Pipeline?** The imputer and scaler learn numbers from the data (medians, means, std). Because they sit inside the pipeline, `pipe.fit(X_train, y_train)` learns those numbers from **training data only**. The test set never influences preprocessing. That is what "leakage-safe" means.

**`class_weight="balanced"`** gives the minority class (non-bipolar) more weight so the model does not just favour the 77% majority. Gradient Boosting has no such option in sklearn, which is fine because its boosting handles this reasonably.

### Step 5 `compute_metrics` and `evaluate_models`
For each model: `fit` on train, then `predict_proba(X_test)[:, 1]` gives the probability of "bipolar". A threshold of 0.5 turns probabilities into class predictions. Metrics:

| Metric | Meaning in one line |
|---|---|
| **AUROC** | Chance a random positive is ranked above a random negative. 0.5 = coin flip, 1.0 = perfect ranking. Threshold-independent. |
| **AUPRC** | Area under precision-recall curve. More informative when classes are imbalanced. |
| **Sensitivity (recall)** | TP / (TP+FN): of all bipolar patients, how many we caught. |
| **Specificity** | TN / (TN+FP): of all non-bipolar, how many we correctly cleared. |
| **Precision** | TP / (TP+FP): when we say bipolar, how often we are right. |
| **F1** | Harmonic mean of precision and recall. |
| **Accuracy** | Overall fraction correct. Can be misleading with imbalance. |
| **Brier score** | Mean squared error of the probabilities. **Lower is better.** Measures calibration quality. |

### Step 6 `circularity_check(df)`
Counts, for every person, how many of the 7 manic symptoms are present, then compares by class:

| Class | min | max | mean |
|---|---|---|---|
| Not bipolar (0) | 0 | 2 | 1.72 |
| Bipolar (1) | 3 | 7 | 4.02 |

The classes **do not overlap**: non-bipolar always has at most 2, bipolar always has at least 3. A hand-written rule "bipolar if at least 3 manic symptoms" gets **100% accuracy with no ML at all**. This is the proof behind the circularity claim.

### Step 7 `permutation_feature_importance`
Shuffles one feature column at a time in the test set and measures how much AUROC drops. A big drop means the model relies on that feature. Repeated 5 times for stability (mean and std reported). Top features are exactly the 7 manic symptoms (drops of about 0.06 to 0.09); every other feature has a drop of about 0.0001, i.e. is ignored.

### Step 8 `missingness_stress_test`
- Randomly sets 10% to 50% of **test** values to NaN (`X_test.mask(mask)`).
- The already-trained pipeline's imputer fills them with **training medians**. No retraining.
- Records AUROC and Brier at each level.
- The same random seed is used for every model, so all models face the same missing pattern (fair comparison).

### `save_plots` (optional `--plots`)
Saves one figure: ROC curves, calibration curves, AUROC vs missingness.

---

## 3. Actual results (seed 42)

**Test-set performance**

| Model | AUROC | AUPRC | Sens. | Spec. | Prec. | F1 | Acc. | Brier |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0003 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0129 |
| Random Forest | 0.9996 | 0.9999 | 1.000 | 0.730 | 0.927 | 0.962 | 0.939 | 0.0767 |

**Robustness (AUROC)**

| Missing % | 0 | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|
| Logistic Regression | 1.000 | 0.950 | 0.915 | 0.865 | 0.824 | 0.774 |
| Random Forest | 0.9996 | 0.950 | 0.920 | 0.873 | 0.831 | 0.784 |
| Gradient Boosting | 1.000 | 0.949 | 0.914 | 0.861 | 0.828 | 0.779 |

Brier at 50% missing: Random Forest 0.171, Gradient Boosting 0.224, Logistic Regression 0.298.

---

## 4. How to interpret the results

1. **Scores are about 100% because of circularity**, not because the models are brilliant. The label is defined by the symptom rule and the inputs are those symptoms, so the model just rediscovers the rule.
2. **Random Forest has lower specificity (0.73)**. I checked: all 57 of its false positives are people with **exactly 2** manic symptoms, which is right at the boundary (the rule is at least 3). The forest averages many trees and `min_samples_leaf=2` smooths the boundary, so probabilities for these borderline people land above 0.5. Its AUROC is still 0.9996, so the **ranking is nearly perfect**; it is the 0.5 **threshold/calibration** that is imperfect.
3. **Calibration (Brier)**: Logistic Regression is best calibrated here (0.0003) because the true rule is almost linear (a count with a threshold), which suits a linear model.
4. **Under missing data all models degrade similarly** (AUROC falls from 1.0 to about 0.78 at 50% missing). Random Forest keeps the best Brier score, meaning its probabilities degrade more gracefully.
5. **Why does missingness hurt so much?** Median imputation of a binary symptom fills a missing value with 0 or 1 (the training median, usually 0), which erases real symptoms and pushes people below the "3 symptoms" boundary.

---

## 5. Likely faculty questions and model answers

**Q1. Is this early prediction?**
No. The data is a cross-sectional symptom checklist with no timestamps, no follow-up and no sensors. It classifies the *current* label. Early prediction needs longitudinal data (for example people with depression who convert to bipolar later). That is my future work.

**Q2. Why is your accuracy 100%? Isn't that overfitting?**
Not overfitting: I evaluate on a held-out test set. It is circularity. The label is defined by at least 3 manic symptoms, and those symptoms are my inputs. My `circularity_check` shows a one-line rule gets 100% accuracy without ML. I report this deliberately instead of claiming a breakthrough.

**Q3. What is data leakage and how did you prevent it?**
Information from outside the training set (or from the answer itself) reaching the model. I prevented it three ways: (a) removed other diagnosis columns and the ID; (b) split before any preprocessing; (c) put imputer and scaler inside the Pipeline so they fit on training data only.

**Q4. Why these three models?**
Logistic Regression = simple, interpretable baseline. Random Forest = non-linear bagging ensemble. Gradient Boosting = non-linear boosting ensemble. They cover linear vs. two ensemble styles, so I can compare complexity against performance (the idea in my reference paper).

**Q5. Why AUROC and also AUPRC and Brier?**
AUROC measures ranking, AUPRC is more reliable under class imbalance, and Brier measures whether predicted probabilities are trustworthy. A model can rank perfectly but still give poorly scaled probabilities (Random Forest here).

**Q6. Why stratified split?**
So train and test keep the same 77.5% / 22.5% class ratio; otherwise a random split could give a test set with an unrepresentative class mix.

**Q7. Why median imputation if there are no missing values?**
The raw CSV has 0 missing cells. The imputer is there for the stress test, where I deliberately create missing values, and so the pipeline is deployment-ready for real records that do have gaps.

**Q8. What is `class_weight="balanced"`?**
It weights each class inversely to its frequency so the minority class (non-bipolar, 22.5%) is not ignored.

**Q9. What is the dataset's limitation regarding prevalence?**
77.5% of records are bipolar, but real-world prevalence is about 1-3%. The data is clinically enriched, so precision and accuracy here would be much lower in the general population (positive predictive value depends on prevalence).

**Q10. Is the missing-data test external validation?**
No. It is a *synthetic* degradation experiment on the same dataset. External validation needs data from a different hospital or population.

**Q11. What does permutation importance tell us?**
The model's performance collapses when manic symptoms are shuffled, and nothing happens when other features are shuffled. So the model uses exactly the DSM-5 manic criteria, which is further evidence of circularity. It is not a discovery of new biomarkers.

**Q12. Could this be used on patients?**
No. It is an academic prototype. It gives no diagnosis. Models in mental health should assist clinicians, never replace a psychiatric interview.

**Q13. What would you do next?**
(1) Compare bipolar vs. major depression only (the real clinical problem: bipolar is often misdiagnosed as unipolar depression). (2) Use longitudinal data (for example NIMH Data Archive, UK Biobank) to predict *future* conversion. (3) Add wearable actigraphy and sleep data for the multimodal part. (4) Do external validation on another site.

**Q14. Why Random Forest has more false positives than the others?**
See section 4, point 2: all its false positives are borderline people with exactly 2 manic symptoms. A ranking metric (AUROC 0.9996) is nearly perfect; only the 0.5 cut-off misplaces those few borderline cases. Tuning the threshold or calibrating the probabilities (Platt/isotonic) would fix it.

**Q15. What does `random_state=42` do?**
Fixes the random number generator so the split, the forest and the missingness masks are identical each time, making results reproducible.

---

## 6. Honest scope statement (use in your closing)

- Done: leakage-safe baseline, 3 models, 8 metrics, calibration, permutation importance, simulated missingness, circularity analysis.
- Not done (and said openly): multimodal sensor data, longitudinal prediction, external validation.

---

## 7. Quick demo script for tomorrow

1. `python bipolar_ml.py --plots`
2. Point at section 1-2 output: "53 features, 5 columns excluded for leakage."
3. Section 4 table: "Near-perfect scores."
4. Section 5: "Here is why: manic count separates the classes completely."
5. Section 7: "Performance falls as data goes missing."
6. Open `results/evaluation_plots.png`.
7. Close with the scope statement above.
