import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv("tracker/data/step2_drop_detection.csv")

# -------------------------------------------------
# Only analyze drop days
# -------------------------------------------------
drop_df = df[df["is_efficiency_drop"]].copy()

# -------------------------------------------------
# Cause deviation scores (relative to baseline)
# -------------------------------------------------
drop_df["dev_completion"] = (
    drop_df["completion_ratio_mean"] - drop_df["completion_ratio"]
)

drop_df["dev_overload"] = (
    drop_df["number_of_tasks"] - drop_df["task_count_mean"]
)

drop_df["dev_planning"] = (
    drop_df["planning_pressure"] - drop_df["planning_pressure_mean"]
)

drop_df["dev_prioritization"] = (
    drop_df["importance_effort_ratio_mean"] - drop_df["importance_effort_ratio"]
)

drop_df["dev_quality"] = (
    drop_df["task_quality_mean_baseline"] - drop_df["task_quality_mean"]
)

# -------------------------------------------------
# Normalize deviations (fair comparison)
# -------------------------------------------------
DEV_COLS = [
    "dev_completion",
    "dev_overload",
    "dev_planning",
    "dev_prioritization",
    "dev_quality",
]

for col in DEV_COLS:
    max_abs = drop_df[col].abs().max()
    if max_abs > 0:
        drop_df[col] = drop_df[col] / max_abs

# -------------------------------------------------
# Rank causes by magnitude
# -------------------------------------------------
drop_df["primary_cause"] = drop_df[DEV_COLS].abs().idxmax(axis=1)

# -------------------------------------------------
# Save results
# -------------------------------------------------
drop_df.to_csv("tracker/data/step3_cause_analysis.csv", index=False)

print("✅ Step 3 completed. Cause analysis saved.")
