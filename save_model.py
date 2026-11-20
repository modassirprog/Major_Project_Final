import joblib
import os
from bipolar_ml import load_data, prepare_features_target, build_models, train_test_split

# Train and save the Logistic Regression pipeline on synthetic data
df = load_data("data/synthetic_future_bipolar.csv")
X, y, _ = prepare_features_target(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
pipe = build_models(42)["Logistic Regression"]
pipe.fit(X_train, y_train)

os.makedirs("results", exist_ok=True)
joblib.dump(pipe, "results/logistic_regression_pipeline.joblib")
print("Saved fitted pipeline -> results/logistic_regression_pipeline.joblib")

# Quick validation on test set
from sklearn.metrics import roc_auc_score
proba = pipe.predict_proba(X_test)[:, 1]
print(f"Test AUROC: {roc_auc_score(y_test, proba):.4f}")
