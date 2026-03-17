import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
INPUT_FILE = Path("tracker/data/step1_baseline_dataset.csv")
OUTPUT_FILE = Path("tracker/data/step2_drop_detection.csv")

# Drop detection sensitivity (lower = more sensitive)
K_OVERALL = 1.5      # For overall efficiency drops
K_DAYTYPE = 1.2      # For day-type-specific drops (more sensitive)

# Minimum data requirements
MIN_BASELINE_DAYS = 3  # Need at least this many days for reliable detection

# -------------------------------------------------
# Load Data
# -------------------------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"📊 Analyzing {len(df)} days for efficiency drops...")

# -------------------------------------------------
# OVERALL EFFICIENCY DROP DETECTION
# -------------------------------------------------
# Standard approach: efficiency below (mean - K*std)
df["efficiency_threshold_overall"] = (
    df["eff_rolling_mean"] - K_OVERALL * df["eff_rolling_std"]
)

df["is_efficiency_drop_overall"] = (
    (df["daily_efficiency"] < df["efficiency_threshold_overall"]) &
    (df["eff_rolling_mean"].notna())  # Only if we have baseline
)

# Drop magnitude (how far below threshold)
df["drop_magnitude_overall"] = np.where(
    df["is_efficiency_drop_overall"],
    df["efficiency_threshold_overall"] - df["daily_efficiency"],
    0.0
)

# -------------------------------------------------
# DAY-TYPE-AWARE DROP DETECTION (NEW!)
# -------------------------------------------------
# This detects when you underperform relative to YOUR baseline for THIS day type
if "eff_daytype_mean" in df.columns and "eff_daytype_std" in df.columns:
    df["efficiency_threshold_daytype"] = (
        df["eff_daytype_mean"] - K_DAYTYPE * df["eff_daytype_std"]
    )
    
    df["is_efficiency_drop_daytype"] = (
        (df["daily_efficiency"] < df["efficiency_threshold_daytype"]) &
        (df["eff_daytype_mean"].notna())
    )
    
    df["drop_magnitude_daytype"] = np.where(
        df["is_efficiency_drop_daytype"],
        df["efficiency_threshold_daytype"] - df["daily_efficiency"],
        0.0
    )
else:
    # Fallback if day-type baselines don't exist
    df["efficiency_threshold_daytype"] = df["efficiency_threshold_overall"]
    df["is_efficiency_drop_daytype"] = df["is_efficiency_drop_overall"]
    df["drop_magnitude_daytype"] = df["drop_magnitude_overall"]

# -------------------------------------------------
# COMBINED DROP DETECTION
# -------------------------------------------------
# A drop is flagged if EITHER overall OR day-type specific threshold is violated
df["is_efficiency_drop"] = (
    df["is_efficiency_drop_overall"] | df["is_efficiency_drop_daytype"]
)

# Use the worse of the two magnitudes
df["drop_magnitude"] = np.maximum(
    df["drop_magnitude_overall"],
    df["drop_magnitude_daytype"]
)

# -------------------------------------------------
# DROP TYPE CLASSIFICATION (NEW!)
# -------------------------------------------------
def classify_drop_type(row):
    """
    Classifies what type of drop this is:
    - overall: Below overall baseline
    - daytype: Below day-type baseline (but not overall)
    - both: Below both baselines
    - none: Not a drop
    """
    if row["is_efficiency_drop_overall"] and row["is_efficiency_drop_daytype"]:
        return "both"
    elif row["is_efficiency_drop_overall"]:
        return "overall"
    elif row["is_efficiency_drop_daytype"]:
        return "daytype"
    else:
        return "none"

df["drop_type"] = df.apply(classify_drop_type, axis=1)

# -------------------------------------------------
# DROP SEVERITY CLASSIFICATION (NEW!)
# -------------------------------------------------
def classify_drop_severity(magnitude, std):
    """
    Classifies severity based on how many standard deviations below baseline.
    
    mild: 1-2 std below
    moderate: 2-3 std below  
    severe: 3+ std below
    """
    if magnitude == 0 or std == 0:
        return "none"
    
    z_score = magnitude / std
    
    if z_score >= 3:
        return "severe"
    elif z_score >= 2:
        return "moderate"
    elif z_score >= 1:
        return "mild"
    else:
        return "none"

df["drop_severity"] = df.apply(
    lambda row: classify_drop_severity(
        row["drop_magnitude"],
        row["eff_rolling_std"] if row["eff_rolling_std"] > 0 else 0.1
    ),
    axis=1
)

# -------------------------------------------------
# TREND-BASED DROP DETECTION (NEW!)
# -------------------------------------------------
# Detect drops happening during declining trends (worse signal)
if "eff_trend_medium" in df.columns:
    df["drop_during_decline"] = (
        df["is_efficiency_drop"] & (df["eff_trend_medium"] < -0.01)
    )
else:
    df["drop_during_decline"] = False

# -------------------------------------------------
# CONSECUTIVE DROP TRACKING (NEW!)
# -------------------------------------------------
# Track streaks of consecutive drops (warning signal)
df["consecutive_drops"] = 0
streak = 0

for i in range(len(df)):
    if df.loc[i, "is_efficiency_drop"]:
        streak += 1
        df.loc[i, "consecutive_drops"] = streak
    else:
        streak = 0

# Mark dangerous streaks (3+ consecutive drops)
df["drop_streak_warning"] = df["consecutive_drops"] >= 3

# -------------------------------------------------
# RECOVERY TRACKING (NEW!)
# -------------------------------------------------
# Days since last drop (recovery indicator)
df["days_since_last_drop"] = 0
last_drop_idx = -1

for i in range(len(df)):
    if df.loc[i, "is_efficiency_drop"]:
        last_drop_idx = i
        df.loc[i, "days_since_last_drop"] = 0
    elif last_drop_idx >= 0:
        df.loc[i, "days_since_last_drop"] = i - last_drop_idx
    else:
        df.loc[i, "days_since_last_drop"] = i  # Days since start

# Recovery success indicator (bounced back above baseline after drop)
df["recovery_success"] = False
for i in range(1, len(df)):
    if (df.loc[i-1, "is_efficiency_drop"] and 
        not df.loc[i, "is_efficiency_drop"] and
        df.loc[i, "daily_efficiency"] > df.loc[i, "eff_rolling_mean"]):
        df.loc[i, "recovery_success"] = True

# -------------------------------------------------
# PERCENTILE-BASED DROP DETECTION (NEW!)
# -------------------------------------------------
# Alternative method: flag bottom 20% of days
if len(df) >= 10:  # Need enough data
    df["efficiency_percentile"] = df["daily_efficiency"].rank(pct=True) * 100
    df["is_bottom_quintile"] = df["efficiency_percentile"] <= 20
else:
    df["efficiency_percentile"] = 50
    df["is_bottom_quintile"] = False

# -------------------------------------------------
# DAY-TYPE PERCENTILE (NEW!)
# -------------------------------------------------
# What percentile is this day within its day type?
if "day_type" in df.columns:
    def compute_daytype_percentile(group):
        group["daytype_efficiency_percentile"] = (
            group["daily_efficiency"].rank(pct=True) * 100
        )
        return group
    
    df = df.groupby("day_type", group_keys=False).apply(compute_daytype_percentile)
    
    # Bottom 25% within day type = underperforming for that day type
    df["is_daytype_underperformer"] = df["daytype_efficiency_percentile"] <= 25
else:
    df["daytype_efficiency_percentile"] = 50
    df["is_daytype_underperformer"] = False

# -------------------------------------------------
# ANOMALY SCORE (NEW!)
# -------------------------------------------------
# Composite score combining multiple signals
def calculate_anomaly_score(row):
    """
    Combines multiple drop signals into single anomaly score (0-10).
    Higher = more concerning.
    """
    score = 0
    
    # Base score from drop magnitude
    if row["drop_magnitude"] > 0:
        score += min(row["drop_magnitude"] * 10, 4)  # Max 4 points
    
    # Severity bonus
    severity_points = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
    score += severity_points.get(row["drop_severity"], 0)
    
    # Consecutive drops penalty
    if row["consecutive_drops"] >= 2:
        score += min(row["consecutive_drops"] * 0.5, 2)  # Max 2 points
    
    # Declining trend penalty
    if row.get("drop_during_decline", False):
        score += 1
    
    return min(score, 10)  # Cap at 10

df["anomaly_score"] = df.apply(calculate_anomaly_score, axis=1)

# -------------------------------------------------
# DROP CONTEXT SUMMARY (NEW!)
# -------------------------------------------------
def generate_drop_context(row):
    """
    Human-readable summary of drop context.
    """
    if not row["is_efficiency_drop"]:
        return "No drop"
    
    context_parts = []
    
    # Severity
    context_parts.append(f"{row['drop_severity'].upper()} drop")
    
    # Type
    if row["drop_type"] == "both":
        context_parts.append("below both overall AND day-type baseline")
    elif row["drop_type"] == "overall":
        context_parts.append("below overall baseline")
    elif row["drop_type"] == "daytype":
        context_parts.append(f"below {row.get('day_type', 'day-type')} baseline")
    
    # Magnitude
    context_parts.append(f"({row['drop_magnitude']:.3f} below threshold)")
    
    # Streak warning
    if row["consecutive_drops"] > 1:
        context_parts.append(f"⚠️ {row['consecutive_drops']} consecutive drops")
    
    # Trend context
    if row.get("drop_during_decline", False):
        context_parts.append("during declining trend")
    
    # Percentile
    if row.get("daytype_efficiency_percentile", 100) <= 10:
        context_parts.append("⚠️ bottom 10% for this day type")
    
    return " | ".join(context_parts)

df["drop_context"] = df.apply(generate_drop_context, axis=1)

# -------------------------------------------------
# EARLY WARNING INDICATORS (NEW!)
# -------------------------------------------------
# Flag days that aren't drops yet but show warning signs
df["early_warning"] = (
    (~df["is_efficiency_drop"]) &  # Not a drop yet
    (
        (df["daily_efficiency"] < df["eff_rolling_mean"]) |  # Below average
        (df.get("eff_trend_short", 0) < -0.02) |  # Declining short-term
        (df.get("efficiency_momentum", 0) < -0.05)  # Negative momentum
    )
)

# -------------------------------------------------
# STATISTICS SUMMARY
# -------------------------------------------------
total_days = len(df)
drop_days = df["is_efficiency_drop"].sum()
drop_rate = drop_days / total_days * 100 if total_days > 0 else 0

severity_counts = df[df["is_efficiency_drop"]]["drop_severity"].value_counts()
max_streak = df["consecutive_drops"].max()

print(f"\n📈 Drop Detection Summary:")
print(f"  • Total days analyzed: {total_days}")
print(f"  • Days with drops: {drop_days} ({drop_rate:.1f}%)")
print(f"  • Severity breakdown: {severity_counts.to_dict()}")
print(f"  • Max consecutive drops: {max_streak}")
print(f"  • Days with early warnings: {df['early_warning'].sum()}")

if "day_type" in df.columns:
    print(f"\n📊 Drops by Day Type:")
    daytype_drops = df[df["is_efficiency_drop"]].groupby("day_type").size()
    for day_type, count in daytype_drops.items():
        total_of_type = (df["day_type"] == day_type).sum()
        rate = count / total_of_type * 100 if total_of_type > 0 else 0
        print(f"  • {day_type}: {count}/{total_of_type} ({rate:.1f}%)")

# -------------------------------------------------
# SAVE ENHANCED DROP DETECTION DATASET
# -------------------------------------------------
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Step 2 completed. Enhanced drop detection saved to {OUTPUT_FILE}")
print(f"\n🔍 New Features Added:")
print("  • Day-type-aware drop detection")
print("  • Drop severity classification (mild/moderate/severe)")
print("  • Drop type classification (overall/daytype/both)")
print("  • Consecutive drop tracking & warnings")
print("  • Recovery tracking & success indicators")
print("  • Percentile-based detection")
print("  • Anomaly scoring (0-10 scale)")
print("  • Trend-context awareness")
print("  • Early warning system")
print("  • Drop context summaries")

# Show example drops
drops_sample = df[df["is_efficiency_drop"]].head(3)
if len(drops_sample) > 0:
    print(f"\n📋 Example Drop Contexts:")
    for idx, row in drops_sample.iterrows():
        print(f"\nDay {row.get('run_id', idx)}:")
        print(f"  {row['drop_context']}")
        print(f"  Anomaly Score: {row['anomaly_score']:.1f}/10")