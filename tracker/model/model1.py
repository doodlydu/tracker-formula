import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
INPUT_FILE = Path("tracker/data/step2_drop_detection.csv")
OUTPUT_FILE = Path("tracker/data/model1_predictions.csv")

# EMA Configuration
ALPHA_DEFAULT = 0.4  # Default smoothing factor
TEST_SIZE = 7  # Last N days for validation

# Day-type specific alphas (learned from patterns)
ALPHA_BY_DAYTYPE = {
    "efficient": 0.35,  # Less smoothing - react faster to Efficient day variations
    "normal": 0.40,     # Balanced
    "chill": 0.45       # More smoothing - Chill days more variable
}

# -------------------------------------------------
# Load Enhanced Dataset
# -------------------------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"📊 Loaded {len(df)} days of efficiency data")

# Ensure sorted by date/run_id
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
elif "run_id" in df.columns:
    df = df.sort_values("run_id").reset_index(drop=True)

# -------------------------------------------------
# BASELINE: Simple EMA (Original Model)
# -------------------------------------------------
def compute_simple_ema(series, alpha=ALPHA_DEFAULT):
    """Traditional exponential moving average."""
    return series.ewm(alpha=alpha, adjust=False).mean()

df["ema_simple"] = compute_simple_ema(df["daily_efficiency"], ALPHA_DEFAULT)

# -------------------------------------------------
# ENHANCED MODEL 1: Day-Type-Aware EMA
# -------------------------------------------------
def compute_daytype_ema(group, alpha_map):
    """Computes EMA with day-type-specific alpha."""
    day_type = group.name.lower() if isinstance(group.name, str) else "normal"
    alpha = alpha_map.get(day_type, ALPHA_DEFAULT)
    group["ema_daytype"] = group["daily_efficiency"].ewm(alpha=alpha, adjust=False).mean()
    return group

if "day_type" in df.columns:
    df = df.groupby("day_type", group_keys=False).apply(
        lambda g: compute_daytype_ema(g, ALPHA_BY_DAYTYPE)
    )
else:
    df["ema_daytype"] = df["ema_simple"]

# -------------------------------------------------
# ENHANCED MODEL 2: Trend-Adjusted EMA
# -------------------------------------------------
def compute_trend_adjusted_ema(df, base_alpha=ALPHA_DEFAULT):
    """
    Adjusts alpha based on trend direction:
    - Increasing trend: Lower alpha (trust the trend)
    - Decreasing trend: Higher alpha (react faster to problems)
    """
    ema_values = []
    current_ema = df["daily_efficiency"].iloc[0]
    
    for i in range(len(df)):
        efficiency = df["daily_efficiency"].iloc[i]
        trend = df.get("eff_trend_short", pd.Series([0]*len(df))).iloc[i]
        
        # Adjust alpha based on trend
        if trend > 0.01:  # Strong positive trend
            alpha = base_alpha * 0.8  # Trust the upward trend
        elif trend < -0.01:  # Strong negative trend
            alpha = base_alpha * 1.2  # React faster to decline
        else:
            alpha = base_alpha
        
        alpha = np.clip(alpha, 0.1, 0.9)  # Keep in reasonable range
        
        # Compute EMA
        current_ema = alpha * efficiency + (1 - alpha) * current_ema
        ema_values.append(current_ema)
    
    return pd.Series(ema_values, index=df.index)

df["ema_trend_adjusted"] = compute_trend_adjusted_ema(df, ALPHA_DEFAULT)

# -------------------------------------------------
# ENHANCED MODEL 3: Weighted EMA (Feature-Based)
# -------------------------------------------------
def compute_weighted_ema(df, base_alpha=ALPHA_DEFAULT):
    """
    Combines multiple signals for prediction:
    - Recent efficiency (EMA)
    - Completion ratio trend
    - Quality trend
    - Momentum indicators
    """
    weights = {
        "ema": 0.5,
        "completion": 0.2,
        "quality": 0.15,
        "momentum": 0.15
    }
    
    # Base EMA
    ema = df["daily_efficiency"].ewm(alpha=base_alpha, adjust=False).mean()
    
    # Completion ratio predictor
    if "completion_ratio_mean" in df.columns:
        completion_pred = df["completion_ratio_mean"] * df["eff_rolling_mean"]
    else:
        completion_pred = ema
    
    # Quality predictor
    if "task_quality_mean_baseline" in df.columns and "eff_rolling_mean" in df.columns:
        quality_pred = (df["task_quality_mean_baseline"] / 5.0) * df["eff_rolling_mean"]
    else:
        quality_pred = ema
    
    # Momentum predictor
    if "efficiency_momentum" in df.columns and "eff_rolling_mean" in df.columns:
        momentum_pred = df["eff_rolling_mean"] + df["efficiency_momentum"] * 0.5
    else:
        momentum_pred = ema
    
    # Weighted combination
    weighted_ema = (
        weights["ema"] * ema +
        weights["completion"] * completion_pred +
        weights["quality"] * quality_pred +
        weights["momentum"] * momentum_pred
    )
    
    return weighted_ema

df["ema_weighted"] = compute_weighted_ema(df, ALPHA_DEFAULT)

# -------------------------------------------------
# ENHANCED MODEL 4: Adaptive Alpha EMA
# -------------------------------------------------
def compute_adaptive_ema(df, base_alpha=ALPHA_DEFAULT):
    """
    Dynamically adjusts alpha based on:
    - Volatility (high volatility = higher alpha)
    - Drop streaks (consecutive drops = higher alpha)
    - Recovery state (recovering = lower alpha)
    """
    ema_values = []
    current_ema = df["daily_efficiency"].iloc[0]
    
    for i in range(len(df)):
        efficiency = df["daily_efficiency"].iloc[i]
        
        # Volatility adjustment
        volatility = df.get("efficiency_volatility", pd.Series([0.05]*len(df))).iloc[i]
        vol_factor = np.clip(volatility / 0.1, 0.8, 1.3)  # Scale based on typical volatility
        
        # Drop streak adjustment (react faster during streaks)
        streak = df.get("consecutive_drops", pd.Series([0]*len(df))).iloc[i]
        streak_factor = 1.0 + (streak * 0.1)  # Increase alpha by 10% per drop in streak
        
        # Recovery adjustment (trust recovery trend)
        recovering = df.get("recovery_success", pd.Series([False]*len(df))).iloc[i]
        recovery_factor = 0.8 if recovering else 1.0
        
        # Compute adaptive alpha
        alpha = base_alpha * vol_factor * streak_factor * recovery_factor
        alpha = np.clip(alpha, 0.1, 0.9)
        
        # Compute EMA
        current_ema = alpha * efficiency + (1 - alpha) * current_ema
        ema_values.append(current_ema)
    
    return pd.Series(ema_values, index=df.index)

df["ema_adaptive"] = compute_adaptive_ema(df, ALPHA_DEFAULT)

# -------------------------------------------------
# NEXT-DAY PREDICTIONS (All Models)
# -------------------------------------------------
predictions = {
    "simple_ema": df["ema_simple"].iloc[-1],
    "daytype_ema": df["ema_daytype"].iloc[-1],
    "trend_adjusted": df["ema_trend_adjusted"].iloc[-1],
    "weighted_ema": df["ema_weighted"].iloc[-1],
    "adaptive_ema": df["ema_adaptive"].iloc[-1]
}

# Ensemble prediction (average of all models)
predictions["ensemble"] = np.mean(list(predictions.values()))

# -------------------------------------------------
# CONFIDENCE INTERVALS
# -------------------------------------------------
def calculate_confidence_interval(df, prediction, window=14):
    """Calculate 80% confidence interval based on recent prediction errors."""
    if len(df) < window:
        window = len(df)
    
    recent_errors = df["daily_efficiency"].iloc[-window:] - df["ema_simple"].iloc[-window:]
    std_error = recent_errors.std()
    
    # 80% CI = ±1.28 standard deviations
    lower = prediction - 1.28 * std_error
    upper = prediction + 1.28 * std_error
    
    return lower, upper

ensemble_lower, ensemble_upper = calculate_confidence_interval(df, predictions["ensemble"])

# -------------------------------------------------
# MODEL VALIDATION (on last N days)
# -------------------------------------------------
def validate_model(actual, predicted, model_name):
    """Calculate validation metrics."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }

if len(df) > TEST_SIZE:
    validation_results = []
    test_actual = df["daily_efficiency"].iloc[-TEST_SIZE:]
    
    models = {
        "Simple EMA": df["ema_simple"].iloc[-TEST_SIZE:],
        "Day-Type EMA": df["ema_daytype"].iloc[-TEST_SIZE:],
        "Trend-Adjusted": df["ema_trend_adjusted"].iloc[-TEST_SIZE:],
        "Weighted EMA": df["ema_weighted"].iloc[-TEST_SIZE:],
        "Adaptive EMA": df["ema_adaptive"].iloc[-TEST_SIZE:]
    }
    
    for model_name, predictions_series in models.items():
        metrics = validate_model(test_actual, predictions_series, model_name)
        validation_results.append(metrics)
    
    validation_df = pd.DataFrame(validation_results)
    best_model = validation_df.loc[validation_df["mae"].idxmin(), "model"]
else:
    validation_df = None
    best_model = "ensemble"

# -------------------------------------------------
# DROP RISK PREDICTION
# -------------------------------------------------
def predict_drop_risk(df, next_day_pred):
    """
    Predicts probability of drop tomorrow based on:
    - Predicted efficiency vs threshold
    - Current trend
    - Recent drop pattern
    """
    risk_score = 0
    
    # Compare to threshold
    if "efficiency_threshold_overall" in df.columns:
        threshold = df["efficiency_threshold_overall"].iloc[-1]
        if next_day_pred < threshold:
            risk_score += 40  # High base risk
        elif next_day_pred < threshold + 0.05:
            risk_score += 20  # Moderate risk (close to threshold)
    
    # Trend risk
    if "eff_trend_short" in df.columns:
        trend = df["eff_trend_short"].iloc[-1]
        if trend < -0.02:
            risk_score += 20  # Strong declining trend
        elif trend < 0:
            risk_score += 10  # Mild declining trend
    
    # Recent drop pattern
    if "consecutive_drops" in df.columns:
        streak = df["consecutive_drops"].iloc[-1]
        if streak > 0:
            risk_score += min(streak * 15, 30)  # Ongoing streak
    
    # Early warning signal
    if df.get("early_warning", pd.Series([False]*len(df))).iloc[-1]:
        risk_score += 10
    
    risk_score = min(risk_score, 100)
    
    # Risk level
    if risk_score >= 60:
        risk_level = "HIGH"
    elif risk_score >= 30:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"
    
    return risk_score, risk_level

drop_risk_score, drop_risk_level = predict_drop_risk(df, predictions["ensemble"])

# -------------------------------------------------
# SAVE PREDICTIONS
# -------------------------------------------------
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

df["ema_ensemble"] = df[["ema_simple", "ema_daytype", "ema_trend_adjusted", 
                          "ema_weighted", "ema_adaptive"]].mean(axis=1)
# -------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------
print("\n" + "="*80)
print("📈 EFFICIENCY FORECASTING REPORT")
print("="*80)

# Recent history
print(f"\n📊 Recent Efficiency History (Last 10 Days):")
recent = df[["daily_efficiency", "ema_simple", "ema_ensemble"]].tail(10)
recent.columns = ["Actual", "Simple EMA", "Ensemble"]
if "run_id" in df.columns:
    recent.index = df["run_id"].tail(10).values
print(recent.round(3).to_string())

# Predictions
print(f"\n🔮 Next-Day Efficiency Predictions:")
print("-" * 50)
for model_name, pred_value in predictions.items():
    print(f"  {model_name:20s}: {pred_value:.3f}")
print("-" * 50)

# Confidence interval
print(f"\n📉 Ensemble Prediction: {predictions['ensemble']:.3f}")
print(f"   80% Confidence Interval: [{ensemble_lower:.3f}, {ensemble_upper:.3f}]")

# Drop risk
print(f"\n⚠️  Drop Risk Assessment:")
print(f"   Risk Score: {drop_risk_score}/100")
print(f"   Risk Level: {drop_risk_level}")

if drop_risk_level == "HIGH":
    print("   🚨 WARNING: High probability of efficiency drop tomorrow!")
    print("   💡 Recommendation: Plan conservatively, focus on high-priority tasks only")
elif drop_risk_level == "MODERATE":
    print("   ⚡ CAUTION: Moderate risk of underperformance")
    print("   💡 Recommendation: Build in buffer time, avoid overcommitting")
else:
    print("   ✅ Low risk: Continue with normal planning")

# Model validation
if validation_df is not None:
    print(f"\n📊 Model Validation (Last {TEST_SIZE} Days):")
    print(validation_df.round(4).to_string(index=False))
    print(f"\n🏆 Best Performing Model: {best_model}")

# Day-type context
if "day_type" in df.columns:
    last_daytype = df["day_type"].iloc[-1]
    print(f"\n📅 Last Day Type: {last_daytype}")
    
    if "eff_daytype_mean" in df.columns:
        daytype_baseline = df["eff_daytype_mean"].iloc[-1]
        print(f"   Your {last_daytype} baseline: {daytype_baseline:.3f}")
        print(f"   Tomorrow's prediction vs baseline: {predictions['ensemble'] - daytype_baseline:+.3f}")

# Trend context
if "eff_trend_medium" in df.columns:
    trend = df["eff_trend_medium"].iloc[-1]
    trend_direction = "improving" if trend > 0.01 else "declining" if trend < -0.01 else "stable"
    print(f"\n📈 Current Trend: {trend_direction} ({trend:+.4f} per day)")

# Streak info
if "consecutive_drops" in df.columns and df["consecutive_drops"].iloc[-1] > 0:
    streak = int(df["consecutive_drops"].iloc[-1])
    print(f"\n🔴 WARNING: Currently in {streak}-day drop streak!")

print("\n" + "="*80)
print(f"✅ Predictions saved to: {OUTPUT_FILE}")
print("="*80 + "\n")

# -------------------------------------------------
# OPTIONAL: Create ensemble column for next steps
# -------------------------------------------------
# Compute ensemble for all historical days
df["ema_ensemble"] = df[["ema_simple", "ema_daytype", "ema_trend_adjusted", 
                          "ema_weighted", "ema_adaptive"]].mean(axis=1)

# Re-save with ensemble
df.to_csv(OUTPUT_FILE, index=False)