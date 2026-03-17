# generate_synthetic_100_days.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

def set_day_params(day_type):
    day_type = day_type.lower()
    params = {
        "efficient": {"gamma": 1.20, "delta": 1.00, "alpha": 0.70, "cap": 1.50, "penalty": 0.80, "completion_bonus": 1.08},
        "normal":    {"gamma": 1.00, "delta": 0.80, "alpha": 0.50, "cap": 1.40, "penalty": 0.90, "completion_bonus": 1.04},
        "chill":     {"gamma": 0.60, "delta": 0.50, "alpha": 0.30, "cap": 1.20, "penalty": 1.00, "completion_bonus": 1.00},
    }
    return params[day_type]

def compute_AR(a, pt, params, quality=1.0, importance=1.0):
    if pt <= 0:
        return 0
    r = a / pt
    alpha, cap, penalty = params["alpha"], params["cap"], params["penalty"]
    if r < 1:
        penalty_eff = penalty + (1 - penalty) * (quality * 0.55 + importance * 0.35)
        return r * penalty_eff
    else:
        return min(1 + alpha * (r - 1), cap)

def shape_ar(ar, params, importance=1.0):
    gamma, delta = params["gamma"], params["delta"]
    if importance < 0.6:
        gamma = gamma * (0.5 + (importance / 0.6) * 0.5)
    return ar ** gamma if ar <= 1 else 1 + delta * (ar - 1)

def compute_task_score(planned, achieved, importance, quality, params):
    ar = compute_AR(achieved, planned, params, quality, importance)
    ar_shaped = shape_ar(ar, params, importance)
    if achieved >= planned:
        ar_shaped *= params["completion_bonus"]
    iq_factor = 0.3 + 0.7 * (0.7 * importance + 0.4 * quality)
    return ar_shaped * iq_factor

def compute_daily_efficiency(task_scores, importance_list, quality_list):
    numerator = sum(task_scores)
    denominator = sum([0.5 * i + 0.5 * q for i, q in zip(importance_list, quality_list)])
    return numerator / denominator if denominator > 0 else 0

# Generate 100 days
print("🔧 Generating 100 days of realistic synthetic data with NEW formula...")
start_date = datetime.now() - timedelta(days=99)
day_types = ["efficient", "normal", "chill"]
rows = []

in_slump, slump_counter = False, 0
in_streak, streak_counter = False, 0

for day_num in range(100):
    current_date = start_date + timedelta(days=day_num)
    day_of_week = current_date.weekday()
    
    # Day type selection (more Efficient/Normal on weekdays, more Chill on weekends)
    if day_of_week < 5:
        day_type = np.random.choice(day_types, p=[0.35, 0.50, 0.15])
    else:
        day_type = np.random.choice(day_types, p=[0.15, 0.35, 0.50])
    
    params = set_day_params(day_type)
    
    # Task count based on day type
    if day_type == "efficient":
        base_tasks = np.random.randint(5, 9)
    elif day_type == "normal":
        base_tasks = np.random.randint(4, 7)
    else:
        base_tasks = np.random.randint(3, 6)
    
    # Monday/Friday adjustments
    if day_of_week == 0:  # Monday
        num_tasks = max(3, base_tasks - 2)
    elif day_of_week == 4:  # Friday
        num_tasks = max(3, base_tasks - 1)
    else:
        num_tasks = base_tasks
    
    # Slump/streak patterns for realism
    if not in_slump and np.random.random() < 0.15:
        in_slump, slump_counter = True, np.random.randint(3, 6)
    if in_slump:
        slump_counter -= 1
        if slump_counter <= 0:
            in_slump = False
    if not in_streak and not in_slump and np.random.random() < 0.20:
        in_streak, streak_counter = True, np.random.randint(3, 7)
    if in_streak:
        streak_counter -= 1
        if streak_counter <= 0:
            in_streak = False
    
    tasks, task_scores, importance_list, quality_list = [], [], [], []
    
    for _ in range(num_tasks):
        planned = np.random.uniform(0.5, 5.0)
        
        # Achievement ratio varies by day type
        base_achievement = {
            "efficient": (0.85, 1.25),
            "normal": (0.70, 1.10),
            "chill": (0.60, 0.95)
        }
        min_a, max_a = base_achievement[day_type]
        
        # Day of week effects
        if day_of_week == 0:  # Monday - lower achievement
            min_a, max_a = min_a * 0.85, max_a * 0.90
        elif day_of_week == 4:  # Friday - variable
            max_a = max_a * (1.1 if np.random.random() < 0.3 else 0.95)
        
        # Pattern effects
        if in_slump:
            min_a, max_a = min_a * 0.75, max_a * 0.80
        if in_streak:
            min_a, max_a = min_a * 1.10, max_a * 1.15
        
        achieved = planned * np.random.uniform(min_a, max_a)
        importance = np.clip(np.random.beta(2, 1.5), 0.2, 1.0)
        quality_base = 0.3 + (achieved/planned/max_a) * 0.6
        quality = np.clip(quality_base + np.random.normal(0, 0.1), 0.3, 1.0)
        
        task_score = compute_task_score(planned, achieved, importance, quality, params)
        
        tasks.append({
            "planned": round(planned, 2),
            "achieved": round(achieved, 2),
            "importance": round(importance, 2),
            "quality": round(quality, 2),
            "score": round(task_score, 3)
        })
        task_scores.append(task_score)
        importance_list.append(importance)
        quality_list.append(quality)
    
    daily_efficiency = compute_daily_efficiency(task_scores, importance_list, quality_list)
    
    row = {
        "run_id": day_num + 1,
        "date": current_date.strftime("%Y-%m-%d"),
        "day_type": day_type.capitalize(),
        "number_of_tasks": num_tasks
    }
    
    # Add tasks (pad to 10)
    for i in range(1, 11):
        if i <= len(tasks):
            row[f"task_{i}_planned"] = tasks[i-1]["planned"]
            row[f"task_{i}_achieved"] = tasks[i-1]["achieved"]
            row[f"task_{i}_importance"] = tasks[i-1]["importance"]
            row[f"task_{i}_quality"] = tasks[i-1]["quality"]
            row[f"task_{i}_score"] = tasks[i-1]["score"]
        else:
            row[f"task_{i}_planned"] = ""
            row[f"task_{i}_achieved"] = ""
            row[f"task_{i}_importance"] = ""
            row[f"task_{i}_quality"] = ""
            row[f"task_{i}_score"] = ""
    
    row["daily_efficiency"] = round(daily_efficiency, 3)
    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# Save to CSV
output_file = "synthetic_efficiency_100_days.csv"
df.to_csv(output_file, index=False)

# Print statistics
print(f"\n✅ Generated 100 days of data saved to: {output_file}")
print(f"\n📊 Dataset Statistics:")
print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
print(f"   Efficiency Range: {df['daily_efficiency'].min():.3f} - {df['daily_efficiency'].max():.3f}")
print(f"   Average Efficiency: {df['daily_efficiency'].mean():.3f} ± {df['daily_efficiency'].std():.3f}")

print(f"\n📈 Day Type Distribution:")
print(df["day_type"].value_counts())

print(f"\n📊 Efficiency by Day Type:")
summary = df.groupby("day_type")["daily_efficiency"].agg(['count', 'mean', 'std', 'min', 'max'])
print(summary.round(3))

# Weekly pattern
df['day_of_week'] = pd.to_datetime(df['date']).dt.day_name()
print(f"\n📅 Average Efficiency by Day of Week:")
weekday_eff = df.groupby('day_of_week')['daily_efficiency'].mean()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for day in weekday_order:
    if day in weekday_eff.index:
        print(f"   {day:10s}: {weekday_eff[day]:.3f}")

print("\n✅ Synthetic data generation complete!")