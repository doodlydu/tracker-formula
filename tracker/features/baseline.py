import pandas as pd
df = pd.read_csv("tracker/data/synthetic_efficiency_dataset_50_days.csv")
WINDOW = 7   # rolling window suitable for ~50 days
# =========================================================
# 1. EFFICIENCY BASELINES
# =========================================================
df["eff_rolling_mean"] = (
    df["daily_efficiency"]
    .rolling(WINDOW, min_periods=3)
    .mean()
)

df["eff_rolling_std"] = (
    df["daily_efficiency"]
    .rolling(WINDOW, min_periods=3)
    .std()
)

# =========================================================
# 2. TASK-LEVEL AGGREGATES
# =========================================================
task_scores     = [f"task_{i}_score" for i in range(1, 11)]
task_planned    = [f"task_{i}_planned" for i in range(1, 11)]
task_achieved   = [f"task_{i}_achieved" for i in range(1, 11)]
task_importance = [f"task_{i}_importance" for i in range(1, 11)]
task_quality    = [f"task_{i}_quality" for i in range(1, 11)]

df["total_planned"]  = df[task_planned].sum(axis=1)
df["total_achieved"] = df[task_achieved].sum(axis=1)

df["completion_ratio"] = (
    df["total_achieved"] / df["total_planned"]
)

df["avg_task_score"] = df[task_scores].mean(axis=1)

# =========================================================
# 3. WORKLOAD BASELINES
# =========================================================
df["task_count_mean"] = (
    df["number_of_tasks"]
    .rolling(WINDOW, min_periods=3)
    .mean()
)

df["task_count_std"] = (
    df["number_of_tasks"]
    .rolling(WINDOW, min_periods=3)
    .std()
)

# =========================================================
# 4. COMPLETION BEHAVIOR BASELINES
# =========================================================
df["completion_ratio_mean"] = (
    df["completion_ratio"]
    .rolling(WINDOW, min_periods=3)
    .mean()
)

df["completion_ratio_std"] = (
    df["completion_ratio"]
    .rolling(WINDOW, min_periods=3)
    .std()
)

# =========================================================
# 5. PRIORITY / IMPORTANCE ALLOCATION BASELINES
# =========================================================
df["importance_weighted_effort"] = (
    df[task_importance] * df[task_achieved]
).sum(axis=1)

df["importance_effort_ratio"] = (
    df["importance_weighted_effort"] / df["total_achieved"]
)

df["importance_effort_ratio_mean"] = (
    df["importance_effort_ratio"]
    .rolling(WINDOW, min_periods=3)
    .mean()
)

# 6. QUALITY / FOCUS BASELINES
df["task_quality_mean"] = df[task_quality].mean(axis=1)
df["task_quality_std"]  = df[task_quality].std(axis=1)
df["task_quality_mean_baseline"] = (
    df["task_quality_mean"]
    .rolling(WINDOW, min_periods=3)
    .mean()
)
# 7. PLANNING AGGRESSIVENESS BASELINES
df["planning_pressure"] = (
    df["total_planned"] / df["total_achieved"]
)
df["planning_pressure_mean"] = (
    df["planning_pressure"]
    .rolling(WINDOW, min_periods=3)
    .mean()
)
# 8. CONSISTENCY / VOLATILITY BASELINE
df["efficiency_volatility"] = (
    df["daily_efficiency"]
    .rolling(WINDOW, min_periods=3)
    .std()
)
df.to_csv("step1_baseline_dataset.csv", index=False)

