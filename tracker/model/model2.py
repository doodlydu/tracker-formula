import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("tracker/data/step3_cause_analysis.csv")
# -------------------------------------------------
# Feature matrix & target
# -------------------------------------------------
FEATURES = [
    "dev_completion",
    "dev_overload",
    "dev_planning",
    "dev_prioritization",
    "dev_quality",
]

X = df[FEATURES].fillna(0)
y = df["daily_efficiency"]

# -------------------------------------------------
# Scale features (important for Ridge)
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# Train model
# -------------------------------------------------
model = Ridge(alpha=1.0)
model.fit(X_scaled, y)

# -------------------------------------------------
# Learned weights
# -------------------------------------------------
weights = pd.Series(
    model.coef_,
    index=FEATURES,
    name="learned_weight"
)

# -------------------------------------------------
# Compute weighted impact per row
# -------------------------------------------------
impact = X_scaled * model.coef_
impact_df = pd.DataFrame(
    impact,
    columns=[f"{f}_impact" for f in FEATURES]
)

df = pd.concat([df.reset_index(drop=True), impact_df], axis=1)

# -------------------------------------------------
# Rank causes by impact
# -------------------------------------------------
impact_cols = impact_df.columns
df["top_cause"] = df[impact_cols].abs().idxmax(axis=1)

# -------------------------------------------------
# Save output
# -------------------------------------------------
df.to_csv("tracker/data/step5_weighted_recommendations.csv", index=False)
print("✅ Step 5 completed.")
print("\nLearned cause weights:")
print(weights.sort_values(key=abs, ascending=False))
