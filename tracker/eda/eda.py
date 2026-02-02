# EDA on the synthetic efficiency dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("synthetic_efficiency_dataset_50_days.csv")

# --------- 1. Trend of daily efficiency ---------
plt.figure()
plt.plot(df["date"], df["daily_efficiency"])
plt.xlabel("Day")
plt.ylabel("Daily Efficiency")
plt.title("Daily Efficiency Trend Over Time")
plt.show()

# --------- 2. Outlier detection (Boxplot) ---------
plt.figure()
plt.boxplot(df["daily_efficiency"].dropna())
plt.ylabel("Daily Efficiency")
plt.title("Outlier Detection for Daily Efficiency")
plt.show()

# --------- 3. Feature Engineering for EDA ---------
# Aggregate task-level features
task_scores = [f"task_{i}_score" for i in range(1, 11)]
task_importance = [f"task_{i}_importance" for i in range(1, 11)]
task_achieved = [f"task_{i}_achieved" for i in range(1, 11)]
task_planned = [f"task_{i}_planned" for i in range(1, 11)]

df["avg_task_score"] = df[task_scores].mean(axis=1)
df["total_tasks_completed"] = df[task_achieved].sum(axis=1)
df["total_tasks_planned"] = df[task_planned].sum(axis=1)
df["completion_ratio"] = df["total_tasks_completed"] / df["total_tasks_planned"]

# --------- 4. Correlation with target ---------
features = [
    "number_of_tasks",
    "avg_task_score",
    "total_tasks_completed",
    "completion_ratio"
]

correlations = df[features + ["daily_efficiency"]].corr()["daily_efficiency"].drop("daily_efficiency")

plt.figure()
plt.bar(correlations.index, correlations.values)
plt.xticks(rotation=30)
plt.ylabel("Correlation")
plt.title("Feature Impact on Daily Efficiency")
plt.show()

# --------- 5. Scatter plots for strongest features ---------
for feature in correlations.index:
    plt.figure()
    plt.scatter(df[feature], df["daily_efficiency"])
    plt.xlabel(feature)
    plt.ylabel("Daily Efficiency")
    plt.title(f"{feature} vs Daily Efficiency")
    plt.show()
