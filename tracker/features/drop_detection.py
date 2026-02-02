import pandas as pd
import numpy as np

# -------------------------------------------------
# STEP 2: Load Step-1 Baseline Dataset
# -------------------------------------------------
df = pd.read_csv("tracker/data/step1_baseline_dataset.csv")

# -------------------------------------------------
# Drop detection parameters
# -------------------------------------------------
K = 1.5  # sensitivity factor

# -------------------------------------------------
# Detect efficiency drops
# -------------------------------------------------
df["efficiency_threshold"] = (
    df["eff_rolling_mean"] - K * df["eff_rolling_std"]
)

df["is_efficiency_drop"] = (
    df["daily_efficiency"] < df["efficiency_threshold"]
)

# -------------------------------------------------
# Drop severity (how bad the drop is)
# -------------------------------------------------
df["drop_magnitude"] = np.where(
    df["is_efficiency_drop"],
    df["efficiency_threshold"] - df["daily_efficiency"],
    0.0
)

# -------------------------------------------------
# Save Step-2 dataset
# -------------------------------------------------
df.to_csv("tracker/data/step2_drop_detection.csv", index=False)

print("✅ Step 2 completed. Drop detection dataset saved.")
