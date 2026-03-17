import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# CONFIGURATION
# =========================================================
INPUT_FILE = Path("synthetic_efficiency_100_days.csv")
OUTPUT_FILE = Path("tracker/data/step1_baseline_dataset.csv")

WINDOW = 7  # rolling window for baselines
MIN_PERIODS = 3  # minimum data points needed for calculation

# =========================================================
# Load Data
# =========================================================
df = pd.read_csv(INPUT_FILE)

# Ensure date column exists and is sorted
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

# =========================================================
# 1. EFFICIENCY BASELINES (OVERALL)
# =========================================================
df["eff_rolling_mean"] = (
    df["daily_efficiency"]
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .mean()
)

df["eff_rolling_std"] = (
    df["daily_efficiency"]
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .std()
    .fillna(0)  # Early days have no std
)

# =========================================================
# 2. DAY-TYPE-AWARE EFFICIENCY BASELINES (NEW!)
# =========================================================
def compute_day_type_baseline(group):
    """
    Computes rolling baseline specific to each day type.
    This captures your 'normal' for Efficient vs Normal vs Chill days separately.
    """
    group = group.sort_values("date") if "date" in group.columns else group
    
    group["eff_daytype_mean"] = (
        group["daily_efficiency"]
        .rolling(WINDOW, min_periods=MIN_PERIODS)
        .mean()
    )
    
    group["eff_daytype_std"] = (
        group["daily_efficiency"]
        .rolling(WINDOW, min_periods=MIN_PERIODS)
        .std()
        .fillna(0)
    )
    
    return group

if "day_type" in df.columns:
    df = df.groupby("day_type", group_keys=False).apply(compute_day_type_baseline)
else:
    # Fallback if no day_type column
    df["eff_daytype_mean"] = df["eff_rolling_mean"]
    df["eff_daytype_std"] = df["eff_rolling_std"]

# =========================================================
# 3. TREND DIRECTION FEATURES (NEW!)
# =========================================================
# Short-term trend (last 3 days)
df["eff_trend_short"] = (
    df["daily_efficiency"]
    .rolling(3, min_periods=2)
    .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0, raw=False)
)

# Medium-term trend (last 7 days)
df["eff_trend_medium"] = (
    df["daily_efficiency"]
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= MIN_PERIODS else 0, raw=False)
)

# Trend classification
def classify_trend(slope, threshold=0.02):
    """Classify trend as improving, declining, or stable"""
    if slope > threshold:
        return "improving"
    elif slope < -threshold:
        return "declining"
    else:
        return "stable"

df["eff_trend_direction"] = df["eff_trend_medium"].apply(classify_trend)

# Days since last improvement
df["days_since_improvement"] = 0
last_peak_idx = 0
for i in range(1, len(df)):
    if df.loc[i, "daily_efficiency"] > df.loc[i-1:i, "daily_efficiency"].max():
        last_peak_idx = i
    df.loc[i, "days_since_improvement"] = i - last_peak_idx

# =========================================================
# 4. TASK-LEVEL AGGREGATES
# =========================================================
# Dynamically detect number of tasks
task_score_cols = [col for col in df.columns if col.startswith("task_") and col.endswith("_score")]
max_tasks = len(task_score_cols)

task_scores     = [f"task_{i}_score" for i in range(1, max_tasks + 1)]
task_planned    = [f"task_{i}_planned" for i in range(1, max_tasks + 1)]
task_achieved   = [f"task_{i}_achieved" for i in range(1, max_tasks + 1)]
task_importance = [f"task_{i}_importance" for i in range(1, max_tasks + 1)]
task_quality    = [f"task_{i}_quality" for i in range(1, max_tasks + 1)]

df["total_planned"]  = df[task_planned].sum(axis=1)
df["total_achieved"] = df[task_achieved].sum(axis=1)

df["completion_ratio"] = (
    df["total_achieved"] / df["total_planned"]
).replace([np.inf, -np.inf], 0).fillna(0)

df["avg_task_score"] = df[task_scores].mean(axis=1)

# Task score variance (measures consistency across tasks)
df["task_score_variance"] = df[task_scores].std(axis=1).fillna(0)

# =========================================================
# 5. WORKLOAD BASELINES
# =========================================================
df["task_count_mean"] = (
    df["number_of_tasks"]
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .mean()
)

df["task_count_std"] = (
    df["number_of_tasks"]
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .std()
    .fillna(0)
)

# Workload intensity (total planned hours)
df["workload_intensity_mean"] = (
    df["total_planned"]
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .mean()
)

# =========================================================
# 6. COMPLETION BEHAVIOR BASELINES
# =========================================================
df["completion_ratio_mean"] = (
    df["completion_ratio"]
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .mean()
)

df["completion_ratio_std"] = (
    df["completion_ratio"]
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .std()
    .fillna(0)
)

# =========================================================
# 7. PRIORITY / IMPORTANCE ALLOCATION BASELINES
# =========================================================
df["importance_weighted_effort"] = (
    df[task_importance] * df[task_achieved]
).sum(axis=1)

df["importance_effort_ratio"] = (
    df["importance_weighted_effort"] / df["total_achieved"]
).replace([np.inf, -np.inf], 0).fillna(0)

df["importance_effort_ratio_mean"] = (
    df["importance_effort_ratio"]
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .mean()
)

df["importance_effort_ratio_std"] = (
    df["importance_effort_ratio"]
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .std()
    .fillna(0)
)

# =========================================================
# 8. QUALITY / FOCUS BASELINES
# =========================================================
df["task_quality_mean"] = df[task_quality].mean(axis=1)
df["task_quality_std"]  = df[task_quality].std(axis=1).fillna(0)

df["task_quality_mean_baseline"] = (
    df["task_quality_mean"]
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .mean()
)

df["task_quality_std_baseline"] = (
    df["task_quality_std"]
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .mean()
)

# Quality-weighted completion (did you complete high-quality work?)
df["quality_weighted_completion"] = (
    (df[task_quality] * df[task_achieved]).sum(axis=1) / df["total_achieved"]
).replace([np.inf, -np.inf], 0).fillna(0)

# =========================================================
# 9. PLANNING AGGRESSIVENESS BASELINES
# =========================================================
df["planning_pressure"] = (
    df["total_planned"] / df["total_achieved"]
).replace([np.inf, -np.inf], 0).fillna(0)

df["planning_pressure_mean"] = (
    df["planning_pressure"]
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .mean()
)

df["planning_pressure_std"] = (
    df["planning_pressure"]
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .std()
    .fillna(0)
)

# Over-planning tendency (how often do you plan more than you can do?)
df["overplanning_frequency"] = (
    (df["planning_pressure"] > 1.2)  # 20% over-planning threshold
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .mean()
)

# =========================================================
# 10. CONSISTENCY / VOLATILITY BASELINE
# =========================================================
df["efficiency_volatility"] = (
    df["daily_efficiency"]
    .rolling(WINDOW, min_periods=MIN_PERIODS)
    .std()
    .fillna(0)
)

# Coefficient of variation (relative volatility)
df["efficiency_cv"] = (
    df["efficiency_volatility"] / df["eff_rolling_mean"]
).replace([np.inf, -np.inf], 0).fillna(0)

# =========================================================
# 11. MOMENTUM INDICATORS (NEW!)
# =========================================================
# Efficiency momentum (current vs baseline)
df["efficiency_momentum"] = (
    df["daily_efficiency"] - df["eff_rolling_mean"]
)

# Consecutive improvement streak
df["improvement_streak"] = 0
streak = 0
for i in range(1, len(df)):
    if df.loc[i, "daily_efficiency"] >= df.loc[i-1, "daily_efficiency"]:
        streak += 1
    else:
        streak = 0
    df.loc[i, "improvement_streak"] = streak

# =========================================================
# 12. DAY-TYPE PERFORMANCE COMPARISON (NEW!)
# =========================================================
if "day_type" in df.columns:
    # How does this day compare to your average for this day type?
    df["daytype_performance_delta"] = (
        df["daily_efficiency"] - df["eff_daytype_mean"]
    )
    
    # Percentile rank within day type
    def compute_percentile_rank(group):
        group["daytype_percentile"] = (
            group["daily_efficiency"].rank(pct=True) * 100
        )
        return group
    
    df = df.groupby("day_type", group_keys=False).apply(compute_percentile_rank)
else:
    df["daytype_performance_delta"] = 0
    df["daytype_percentile"] = 50

# =========================================================
# SAVE OUTPUT
# =========================================================
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("✅ Step 1 completed: Enhanced baseline dataset saved.")
print(f"\n📊 New Features Added:")
print("  • Day-type-aware baselines (eff_daytype_mean, eff_daytype_std)")
print("  • Trend direction features (eff_trend_short, eff_trend_medium, eff_trend_direction)")
print("  • Days since improvement tracker")
print("  • Task score variance (consistency measure)")
print("  • Quality-weighted completion")
print("  • Over-planning frequency")
print("  • Efficiency coefficient of variation")
print("  • Momentum indicators (efficiency_momentum, improvement_streak)")
print("  • Day-type performance comparison (daytype_performance_delta, daytype_percentile)")
print(f"\n📈 Total features created: {len(df.columns)}")