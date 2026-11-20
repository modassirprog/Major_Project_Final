import numpy as np
import pandas as pd

np.random.seed(42)
n = 3750

age = np.random.normal(32, 10, n).clip(18, 70).round(1)
sex = np.random.binomial(1, 0.52, n)
prior_dep = np.random.poisson(1.2, n).clip(0, 8)
prior_anx = np.random.poisson(0.8, n).clip(0, 6)
family_hist = np.random.binomial(1, 0.22, n)
sleep_dist = np.random.beta(2.5, 1.2, n) * 10
stress_score = np.random.beta(2, 1.5, n) * 10
cog_impair = np.random.beta(1.8, 2.2, n) * 10
substance_use = np.random.poisson(0.9, n).clip(0, 5)
social_support = np.random.beta(3, 1.5, n) * 10
mood_instability = np.random.beta(2.5, 1.2, n) * 10
med_adherence = np.clip(np.random.normal(0.65, 0.18, n), 0, 1).round(2)
energy_level = np.clip(np.random.normal(5.5, 1.8, n), 0, 10)
social_withdrawal = np.clip(np.random.normal(3.5, 2.2, n), 0, 10)

logit = (
    -5.5
    + 0.04 * age
    + 0.25 * family_hist
    + 0.18 * prior_dep
    + 0.15 * prior_anx
    + 0.22 * sleep_dist
    + 0.20 * stress_score
    + 0.12 * cog_impair
    + 0.10 * substance_use
    + 0.35 * mood_instability
    - 0.08 * social_support
    + 0.08 * social_withdrawal
    - 0.10 * med_adherence
    + np.random.normal(0, 0.6, n)
)
prob = 1 / (1 + np.exp(-logit))
threshold = np.percentile(prob, 87)
future_bipolar_6mo = (prob >= threshold).astype(int)
flip_mask = np.random.binomial(1, 0.03, n).astype(bool)
future_bipolar_6mo[flip_mask] = 1 - future_bipolar_6mo[flip_mask]

df = pd.DataFrame({
    "ID": np.arange(1, n+1),
    "Age": age,
    "Sex": sex,
    "Prior_depression_episodes": prior_dep,
    "Prior_anxiety_episodes": prior_anx,
    "Family_history_bipolar": family_hist,
    "Sleep_disturbance_score": sleep_dist.round(2),
    "Stress_score": stress_score.round(2),
    "Cognitive_impairment_score": cog_impair.round(2),
    "Substance_use_frequency": substance_use,
    "Social_support_score": social_support.round(2),
    "Baseline_mood_instability": mood_instability.round(2),
    "Medication_adherence": med_adherence,
    "Baseline_energy_level": energy_level.round(2),
    "Baseline_social_withdrawal": social_withdrawal.round(2),
    "Future_bipolar_6mo": future_bipolar_6mo,
})

missing_mask = np.random.random(df.shape) < 0.065
# Do not blank out target column (needs to stay intact for evaluation)
missing_mask[:, df.columns.get_loc("Future_bipolar_6mo")] = False
missing_mask[:, df.columns.get_loc("ID")] = False
df_masked = df.mask(missing_mask)
extra_missing_cols = ["Social_support_score", "Medication_adherence", "Baseline_energy_level"]
for col in extra_missing_cols:
    extra = np.random.random(len(df_masked)) < 0.05
    df_masked.loc[extra, col] = np.nan

# Do NOT drop rows; pipeline handles missing predictor values
# (Keep full 3750 rows for consistency with request)

out_path = "data/synthetic_future_bipolar.csv"
df_masked.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(f"Rows: {len(df_masked)}, Columns: {len(df_masked.columns)}")
print(f"Positive rate: {df_masked['Future_bipolar_6mo'].mean():.2%}")
print(f"Total missing: {int(df_masked.isna().sum().sum())}")
print(f"Columns: {list(df_masked.columns)}")
