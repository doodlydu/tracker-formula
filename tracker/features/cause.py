import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
INPUT_FILE = Path("tracker/data/step2_drop_detection.csv")
OUTPUT_FILE = Path("tracker/data/step3_cause_analysis.csv")

# -------------------------------------------------
# Load data
# -------------------------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

# -------------------------------------------------
# Only analyze drop days
# -------------------------------------------------
drop_df = df[df["is_efficiency_drop"]].copy()

if len(drop_df) == 0:
    print("⚠️  No efficiency drops detected in dataset. Skipping cause analysis.")
    # Save empty file to maintain pipeline
    drop_df.to_csv(OUTPUT_FILE, index=False)
    exit(0)

print(f"📉 Analyzing {len(drop_df)} efficiency drop days...")

# -------------------------------------------------
# CAUSE 1: COMPLETION RATIO DEVIATION
# -------------------------------------------------
# Use day-type-aware baseline if available, otherwise fall back to overall
baseline_col = "completion_ratio_mean"

drop_df["dev_completion_raw"] = (
    drop_df["completion_ratio"] - drop_df[baseline_col]
)

# Z-score (standardized deviation)
drop_df["dev_completion"] = np.where(
    drop_df["completion_ratio_std"] > 0,
    drop_df["dev_completion_raw"] / drop_df["completion_ratio_std"],
    drop_df["dev_completion_raw"]  # Fallback if std is 0
)

# -------------------------------------------------
# CAUSE 2: WORKLOAD OVERLOAD
# -------------------------------------------------
drop_df["dev_overload_raw"] = (
    drop_df["number_of_tasks"] - drop_df["task_count_mean"]
)

drop_df["dev_overload"] = np.where(
    drop_df["task_count_std"] > 0,
    drop_df["dev_overload_raw"] / drop_df["task_count_std"],
    drop_df["dev_overload_raw"]
)

# -------------------------------------------------
# CAUSE 3: PLANNING PRESSURE (Over-ambitious targets)
# -------------------------------------------------
drop_df["dev_planning_raw"] = (
    drop_df["planning_pressure"] - drop_df["planning_pressure_mean"]
)

drop_df["dev_planning"] = np.where(
    drop_df["planning_pressure_std"] > 0,
    drop_df["dev_planning_raw"] / drop_df["planning_pressure_std"],
    drop_df["dev_planning_raw"]
)

# -------------------------------------------------
# CAUSE 4: PRIORITIZATION (Low-importance work)
# -------------------------------------------------
drop_df["dev_prioritization_raw"] = (
    drop_df["importance_effort_ratio_mean"] - drop_df["importance_effort_ratio"]
)

drop_df["dev_prioritization"] = np.where(
    drop_df["importance_effort_ratio_std"] > 0,
    drop_df["dev_prioritization_raw"] / drop_df["importance_effort_ratio_std"],
    drop_df["dev_prioritization_raw"]
)

# -------------------------------------------------
# CAUSE 5: QUALITY DROP
# -------------------------------------------------
drop_df["dev_quality_raw"] = (
    drop_df["task_quality_mean_baseline"] - drop_df["task_quality_mean"]
)

drop_df["dev_quality"] = np.where(
    drop_df["task_quality_std_baseline"] > 0,
    drop_df["dev_quality_raw"] / drop_df["task_quality_std_baseline"],
    drop_df["dev_quality_raw"]
)

# -------------------------------------------------
# CAUSE 6: INCONSISTENT TASK PERFORMANCE (NEW!)
# -------------------------------------------------
# High variance in task scores = some tasks went well, others tanked
if "task_score_variance" in drop_df.columns:
    # Compare to personal average variance
    avg_variance = df["task_score_variance"].mean()
    drop_df["dev_inconsistency_raw"] = (
        drop_df["task_score_variance"] - avg_variance
    )
    
    std_variance = df["task_score_variance"].std()
    drop_df["dev_inconsistency"] = np.where(
        std_variance > 0,
        drop_df["dev_inconsistency_raw"] / std_variance,
        drop_df["dev_inconsistency_raw"]
    )
else:
    drop_df["dev_inconsistency"] = 0

# -------------------------------------------------
# CAUSE 7: WORKLOAD INTENSITY (NEW!)
# -------------------------------------------------
# Total hours planned (not just task count)
if "workload_intensity_mean" in drop_df.columns:
    drop_df["dev_intensity_raw"] = (
        drop_df["total_planned"] - drop_df["workload_intensity_mean"]
    )
    
    # Standardize
    intensity_std = (df["total_planned"].rolling(7, min_periods=3).std().mean())
    drop_df["dev_intensity"] = np.where(
        intensity_std > 0,
        drop_df["dev_intensity_raw"] / intensity_std,
        drop_df["dev_intensity_raw"]
    )
else:
    drop_df["dev_intensity"] = 0

# -------------------------------------------------
# CAUSE 8: NEGATIVE MOMENTUM (NEW!)
# -------------------------------------------------
# Already on a declining trend before this drop
if "eff_trend_medium" in drop_df.columns:
    drop_df["dev_momentum"] = -drop_df["eff_trend_medium"]  # Negative trend = bad
else:
    drop_df["dev_momentum"] = 0

# -------------------------------------------------
# CAUSE 9: DAY-TYPE UNDERPERFORMANCE (NEW!)
# -------------------------------------------------
# Performing poorly relative to your baseline for THIS day type
if "daytype_performance_delta" in drop_df.columns:
    drop_df["dev_daytype"] = -drop_df["daytype_performance_delta"]  # Negative delta = underperformance
else:
    drop_df["dev_daytype"] = 0

# -------------------------------------------------
# Normalize all deviations for fair comparison
# -------------------------------------------------
DEV_COLS = [
    "dev_completion",
    "dev_overload",
    "dev_planning",
    "dev_prioritization",
    "dev_quality",
    "dev_inconsistency",
    "dev_intensity",
    "dev_momentum",
    "dev_daytype"
]

# Remove any that don't exist (for backward compatibility)
DEV_COLS = [col for col in DEV_COLS if col in drop_df.columns]

# Normalize to [0, 1] scale for comparison
for col in DEV_COLS:
    max_abs = drop_df[col].abs().max()
    if max_abs > 0:
        drop_df[f"{col}_normalized"] = drop_df[col].abs() / max_abs
    else:
        drop_df[f"{col}_normalized"] = 0

# -------------------------------------------------
# Multi-Cause Analysis
# -------------------------------------------------
NORMALIZED_COLS = [f"{col}_normalized" for col in DEV_COLS]

# Primary cause (largest deviation)
drop_df["primary_cause"] = drop_df[DEV_COLS].abs().idxmax(axis=1)

# Secondary cause (second largest)
def get_secondary_cause(row):
    values = row[DEV_COLS].abs().sort_values(ascending=False)
    return values.index[1] if len(values) > 1 else None

drop_df["secondary_cause"] = drop_df.apply(get_secondary_cause, axis=1)

# Contributing causes (all causes > 0.3 normalized threshold)
def get_contributing_causes(row):
    causes = []
    for col in DEV_COLS:
        if row[f"{col}_normalized"] > 0.3:  # 30% threshold
            causes.append(col.replace("dev_", ""))
    return ", ".join(causes) if causes else "none"

drop_df["contributing_causes"] = drop_df.apply(get_contributing_causes, axis=1)

# Cause count (how many issues contributed?)
drop_df["cause_count"] = drop_df.apply(
    lambda row: sum(row[f"{col}_normalized"] > 0.3 for col in DEV_COLS),
    axis=1
)

# -------------------------------------------------
# Severity Scoring
# -------------------------------------------------
# Overall severity = weighted sum of normalized deviations
drop_df["overall_severity"] = drop_df[NORMALIZED_COLS].sum(axis=1)

# Severity classification
def classify_severity(severity):
    if severity > 2.0:
        return "severe"
    elif severity > 1.0:
        return "moderate"
    else:
        return "mild"

drop_df["severity_level"] = drop_df["overall_severity"].apply(classify_severity)

# -------------------------------------------------
# Cause Interpretation (human-readable)
# -------------------------------------------------
CAUSE_NAMES = {
    "dev_completion": "Low Completion Rate",
    "dev_overload": "Task Overload",
    "dev_planning": "Over-Ambitious Planning",
    "dev_prioritization": "Poor Prioritization",
    "dev_quality": "Quality Drop",
    "dev_inconsistency": "Inconsistent Performance",
    "dev_intensity": "High Workload Intensity",
    "dev_momentum": "Declining Trend",
    "dev_daytype": "Day-Type Underperformance"
}

drop_df["primary_cause_name"] = drop_df["primary_cause"].map(CAUSE_NAMES)
drop_df["secondary_cause_name"] = drop_df["secondary_cause"].map(CAUSE_NAMES)

# -------------------------------------------------
# Save results
# -------------------------------------------------
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
drop_df.to_csv(OUTPUT_FILE, index=False)

print("✅ Step 3 completed. Enhanced cause analysis saved.")
print(f"\n📊 Cause Analysis Summary:")
print(f"  • Total drop days analyzed: {len(drop_df)}")
print(f"\n🔍 Primary Causes Distribution:")
print(drop_df["primary_cause_name"].value_counts())
print(f"\n⚠️  Severity Distribution:")
print(drop_df["severity_level"].value_counts())
print(f"\n🎯 Average causes per drop: {drop_df['cause_count'].mean():.1f}")

# Show example drop with full analysis
if len(drop_df) > 0:
    print(f"\n📋 Example Drop Analysis (Day {drop_df.iloc[0]['run_id']}):")
    example = drop_df.iloc[0]
    print(f"  Primary Cause: {example['primary_cause_name']}")
    print(f"  Secondary Cause: {example['secondary_cause_name']}")
    print(f"  All Contributing: {example['contributing_causes']}")
    print(f"  Severity: {example['severity_level']} ({example['overall_severity']:.2f})")