# EDA on the synthetic efficiency dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# File path handling
DATA_FILE = Path("synthetic_efficiency_100_days.csv")
OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Check if file exists
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

# Parse date if present
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

# --------- 1. Trend of daily efficiency ---------
plt.figure(figsize=(10, 6))
plt.plot(df["date"] if "date" in df.columns else range(len(df)), 
         df["daily_efficiency"], marker='o')
plt.xlabel("Day")
plt.ylabel("Daily Efficiency")
plt.title("Daily Efficiency Trend Over Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "1_efficiency_trend.png", dpi=150)
plt.show()

# --------- 2. Outlier detection (Boxplot) ---------
plt.figure(figsize=(8, 6))
plt.boxplot(df["daily_efficiency"].dropna())
plt.ylabel("Daily Efficiency")
plt.title("Outlier Detection for Daily Efficiency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "2_outlier_boxplot.png", dpi=150)
plt.show()

# --------- 3. Feature Engineering for EDA ---------
# Dynamically detect max tasks
task_cols = [col for col in df.columns if col.startswith("task_") and col.endswith("_score")]
max_tasks = len(task_cols)

task_scores = [f"task_{i}_score" for i in range(1, max_tasks + 1)]
task_importance = [f"task_{i}_importance" for i in range(1, max_tasks + 1)]
task_achieved = [f"task_{i}_achieved" for i in range(1, max_tasks + 1)]
task_planned = [f"task_{i}_planned" for i in range(1, max_tasks + 1)]

df["avg_task_score"] = df[task_scores].mean(axis=1)
df["total_tasks_completed"] = df[task_achieved].sum(axis=1)
df["total_tasks_planned"] = df[task_planned].sum(axis=1)
df["completion_ratio"] = df["total_tasks_completed"] / df["total_tasks_planned"]

# --------- 4. Day Type Analysis (NEW) ---------
if "day_type" in df.columns:
    plt.figure(figsize=(10, 6))
    day_type_stats = df.groupby("day_type")["daily_efficiency"].agg(['mean', 'std'])
    day_type_stats.plot(kind='bar', y='mean', yerr='std', 
                        capsize=4, rot=0, legend=False)
    plt.xlabel("Day Type")
    plt.ylabel("Average Daily Efficiency")
    plt.title("Daily Efficiency by Day Type")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "3_day_type_comparison.png", dpi=150)
    plt.show()

# --------- 5. Correlation with target ---------
features = [
    "number_of_tasks",
    "avg_task_score",
    "total_tasks_completed",
    "completion_ratio"
]

correlations = df[features + ["daily_efficiency"]].corr()["daily_efficiency"].drop("daily_efficiency")

plt.figure(figsize=(10, 6))
correlations.sort_values().plot(kind='barh')
plt.xlabel("Correlation with Daily Efficiency")
plt.title("Feature Impact on Daily Efficiency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "4_feature_correlations.png", dpi=150)
plt.show()

# Save correlation table
correlations.to_csv(OUTPUT_DIR / "correlations.csv")

# --------- 6. Scatter plots for strongest features ---------
top_n = 3
top_features = correlations.abs().nlargest(top_n).index

for feature in top_features:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[feature], df["daily_efficiency"], alpha=0.6)
    plt.xlabel(feature)
    plt.ylabel("Daily Efficiency")
    plt.title(f"{feature} vs Daily Efficiency (r={correlations[feature]:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"5_scatter_{feature}.png", dpi=150)
    plt.show()

# --------- 7. Summary Statistics ---------
summary = df[["daily_efficiency"] + features].describe()
summary.to_csv(OUTPUT_DIR / "summary_statistics.csv")

print(f"\n✅ EDA complete. Outputs saved to '{OUTPUT_DIR}/'")
print(f"\nSummary Statistics:\n{summary}")