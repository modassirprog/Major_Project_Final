import joblib
import pandas as pd
import numpy as np

# 1. Load the saved fitted pipeline (or train and save it first)
# If you haven't saved it yet, uncomment the training block below:
# ------------------------------------------------------------------
# import os
# from bipolar_ml import load_data, prepare_features_target, build_models, train_test_split
# df = load_data("data/synthetic_future_bipolar.csv")
# X, y, _ = prepare_features_target(df)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
# pipe = build_models(42)["Logistic Regression"]
# pipe.fit(X_train, y_train)
# joblib.dump(pipe, "results/logistic_regression_pipeline.joblib")
# ------------------------------------------------------------------

pipe = joblib.load("results/logistic_regression_pipeline.joblib")
print("Loaded fitted pipeline:", type(pipe))

# 2. Create a new sample with the SAME predictor columns (no target)
# The synthetic dataset predictors (after exclusions) are:
new_sample = pd.DataFrame([{
    "Age": 29.0,
    "Sex": 1,
    "Prior_depression_episodes": 0,
    "Prior_anxiety_episodes": 0,
    "Family_history_bipolar": 0,
    "Sleep_disturbance_score": 6.5,
    "Stress_score": 4.0,
    "Cognitive_impairment_score": 7.0,
    "Substance_use_frequency": 2,
    "Social_support_score": 2.0,
    "Baseline_mood_instability": 2.5,
    "Medication_adherence": 0.3,
    "Baseline_energy_level": 5.0,
    "Baseline_social_withdrawal": 8.0,
}])

# The pipeline handles missing values and scaling automatically.
proba = pipe.predict_proba(new_sample)[0, 1]
print(f"Predicted probability of future bipolar (6mo): {proba:.4f}")
print(f"Risk interpretation: {'High' if proba > 0.5 else 'Low-Medium'}")
