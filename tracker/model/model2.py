import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
INPUT_FILE = Path("tracker/data/step2_drop_detection.csv")
OUTPUT_FILE = Path("tracker/data/model2_predictions.csv")
WEIGHTS_FILE = Path("tracker/data/model2_feature_weights.csv")

# Model hyperparameters
RIDGE_ALPHA = 1.0
LASSO_ALPHA = 0.1
ELASTIC_ALPHA = 0.5
ELASTIC_L1_RATIO = 0.5

# Validation
N_SPLITS = 5  # Time series cross-validation folds
TEST_SIZE = 7  # Last N days for final test

# -------------------------------------------------
# Load Enhanced Dataset
# -------------------------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"📊 Loaded {len(df)} days of efficiency data")

# Ensure sorted
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
elif "run_id" in df.columns:
    df = df.sort_values("run_id").reset_index(drop=True)

# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------

print("\n🔧 Engineering features for ML models...")

# Lag features (yesterday's efficiency)
df["efficiency_lag1"] = df["daily_efficiency"].shift(1)
df["efficiency_lag2"] = df["daily_efficiency"].shift(2)
df["efficiency_lag3"] = df["daily_efficiency"].shift(3)

# Rolling statistics (last 3 days)
df["efficiency_rolling_mean_3d"] = df["daily_efficiency"].rolling(3, min_periods=1).mean()
df["efficiency_rolling_std_3d"] = df["daily_efficiency"].rolling(3, min_periods=1).std().fillna(0)

# Day of week effects (if date available)
if "date" in df.columns:
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
else:
    df["day_of_week"] = 0
    df["is_monday"] = 0
    df["is_friday"] = 0
    df["is_weekend"] = 0

# Task complexity features
df["avg_task_duration"] = df["total_planned"] / df["number_of_tasks"]
df["task_completion_rate"] = df["total_achieved"] / df["total_planned"]

# Interaction features (capturing complex relationships)
df["completion_x_quality"] = df.get("completion_ratio", 0) * df.get("task_quality_mean", 0)
df["workload_x_importance"] = df.get("total_planned", 0) * df.get("importance_effort_ratio", 0)

# Momentum-based features
if "efficiency_momentum" in df.columns:
    df["momentum_squared"] = df["efficiency_momentum"] ** 2
    df["momentum_direction"] = np.sign(df["efficiency_momentum"])
else:
    df["momentum_squared"] = 0
    df["momentum_direction"] = 0

# Streak features
if "improvement_streak" in df.columns:
    df["streak_squared"] = df["improvement_streak"] ** 2
else:
    df["streak_squared"] = 0

# Day-type encoding
if "day_type" in df.columns:
    df["is_efficient_day"] = (df["day_type"].str.lower() == "efficient").astype(int)
    df["is_normal_day"] = (df["day_type"].str.lower() == "normal").astype(int)
    df["is_chill_day"] = (df["day_type"].str.lower() == "chill").astype(int)
else:
    df["is_efficient_day"] = 0
    df["is_normal_day"] = 1
    df["is_chill_day"] = 0

# -------------------------------------------------
# FEATURE SELECTION
# -------------------------------------------------

# Core predictive features
FEATURE_SETS = {
    "lag_features": [
        "efficiency_lag1",
        "efficiency_lag2", 
        "efficiency_lag3",
        "efficiency_rolling_mean_3d",
        "efficiency_rolling_std_3d"
    ],
    
    "baseline_features": [
        "eff_rolling_mean",
        "eff_rolling_std",
        "eff_daytype_mean",
        "eff_daytype_std"
    ],
    
    "trend_features": [
        "eff_trend_short",
        "eff_trend_medium",
        "efficiency_momentum",
        "momentum_squared",
        "momentum_direction"
    ],
    
    "workload_features": [
        "number_of_tasks",
        "total_planned",
        "total_achieved",
        "completion_ratio",
        "avg_task_duration",
        "task_completion_rate",
        "planning_pressure"
    ],
    
    "quality_features": [
        "task_quality_mean",
        "task_quality_std",
        "avg_task_score",
        "task_score_variance"
    ],
    
    "priority_features": [
        "importance_effort_ratio",
        "importance_weighted_effort"
    ],
    
    "streak_features": [
        "consecutive_drops",
        "improvement_streak",
        "streak_squared",
        "days_since_improvement"
    ],
    
    "daytype_features": [
        "is_efficient_day",
        "is_normal_day",
        "is_chill_day"
    ],
    
    "temporal_features": [
        "day_of_week",
        "is_monday",
        "is_friday",
        "is_weekend"
    ],
    
    "interaction_features": [
        "completion_x_quality",
        "workload_x_importance"
    ]
}

# Combine all features
ALL_FEATURES = []
for feature_list in FEATURE_SETS.values():
    ALL_FEATURES.extend(feature_list)

# Filter to only existing columns
AVAILABLE_FEATURES = [f for f in ALL_FEATURES if f in df.columns]

print(f"   Available features: {len(AVAILABLE_FEATURES)}")

# Remove rows with NaN in features (due to lagging)
df_clean = df.dropna(subset=AVAILABLE_FEATURES + ["daily_efficiency"]).copy()

print(f"   Clean dataset: {len(df_clean)} days (removed {len(df) - len(df_clean)} with NaN)")

# -------------------------------------------------
# TRAIN/TEST SPLIT (Time-based)
# -------------------------------------------------

if len(df_clean) < TEST_SIZE + 10:
    raise ValueError(f"Not enough data. Need at least {TEST_SIZE + 10} days, have {len(df_clean)}")

train_df = df_clean.iloc[:-TEST_SIZE].copy()
test_df = df_clean.iloc[-TEST_SIZE:].copy()

X_train = train_df[AVAILABLE_FEATURES]
y_train = train_df["daily_efficiency"]

X_test = test_df[AVAILABLE_FEATURES]
y_test = test_df["daily_efficiency"]

print(f"\n📊 Dataset Split:")
print(f"   Training: {len(train_df)} days")
print(f"   Testing: {len(test_df)} days")

# -------------------------------------------------
# FEATURE SCALING
# -------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------

print(f"\n🤖 Training ML models...")

models = {
    "Ridge": Ridge(alpha=RIDGE_ALPHA),
    "Lasso": Lasso(alpha=LASSO_ALPHA, max_iter=10000),
    "ElasticNet": ElasticNet(alpha=ELASTIC_ALPHA, l1_ratio=ELASTIC_L1_RATIO, max_iter=10000),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
}

trained_models = {}
model_performance = []

for name, model in models.items():
    print(f"   Training {name}...", end=" ")
    
    # Train
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    
    # Cross-validation on training set
    tscv = TimeSeriesSplit(n_splits=min(N_SPLITS, len(train_df) // 10))
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                 cv=tscv, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    
    # Test set performance
    y_pred = model.predict(X_test_scaled)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    test_mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    model_performance.append({
        "model": name,
        "cv_mae": cv_mae,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "test_mape": test_mape
    })
    
    print(f"✓ (Test MAE: {test_mae:.4f})")

performance_df = pd.DataFrame(model_performance)
best_model_name = performance_df.loc[performance_df["test_mae"].idxmin(), "model"]
best_model = trained_models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")

# -------------------------------------------------
# FEATURE IMPORTANCE ANALYSIS
# -------------------------------------------------

print(f"\n🔍 Analyzing feature importance...")

def get_feature_importance(model, feature_names, model_type):
    """Extract feature importance from different model types."""
    if model_type in ["Ridge", "Lasso", "ElasticNet"]:
        # Linear models: use coefficients
        importance = np.abs(model.coef_)
    elif model_type in ["RandomForest", "GradientBoosting"]:
        # Tree models: use feature_importances_
        importance = model.feature_importances_
    else:
        importance = np.zeros(len(feature_names))
    
    return pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
        "abs_importance": np.abs(importance)
    }).sort_values("abs_importance", ascending=False)

# Get importance from best model
importance_df = get_feature_importance(best_model, AVAILABLE_FEATURES, best_model_name)

# Also get importance from Ridge (for interpretability)
ridge_importance = get_feature_importance(
    trained_models["Ridge"], 
    AVAILABLE_FEATURES, 
    "Ridge"
)

# Categorize features by type
def categorize_feature(feature_name):
    for category, features in FEATURE_SETS.items():
        if feature_name in features:
            return category
    return "other"

importance_df["category"] = importance_df["feature"].apply(categorize_feature)

# -------------------------------------------------
# ENSEMBLE PREDICTION
# -------------------------------------------------

print(f"\n🔮 Generating ensemble predictions...")

# Weighted ensemble (weight by inverse MAE)
weights = {}
total_inv_mae = 0

for perf in model_performance:
    inv_mae = 1 / (perf["test_mae"] + 0.001)  # Add small constant to avoid division by zero
    weights[perf["model"]] = inv_mae
    total_inv_mae += inv_mae

# Normalize weights
for model_name in weights:
    weights[model_name] /= total_inv_mae

# Make predictions with all models
all_predictions = {}
for name, model in trained_models.items():
    all_predictions[name] = model.predict(X_test_scaled)

# Ensemble prediction
ensemble_pred = np.zeros(len(X_test))
for name, preds in all_predictions.items():
    ensemble_pred += weights[name] * preds

# Ensemble metrics
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
ensemble_r2 = r2_score(y_test, ensemble_pred)
ensemble_mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100

# -------------------------------------------------
# NEXT-DAY PREDICTION
# -------------------------------------------------

# Prepare features for tomorrow
latest_features = df_clean.iloc[-1:][AVAILABLE_FEATURES]
latest_scaled = scaler.transform(latest_features)

# Predict with all models
next_day_predictions = {}
for name, model in trained_models.items():
    next_day_predictions[name] = model.predict(latest_scaled)[0]

# Ensemble prediction for tomorrow
next_day_ensemble = sum(weights[name] * pred for name, pred in next_day_predictions.items())

# Prediction interval (based on test set residuals)
residuals = y_test - ensemble_pred
residual_std = np.std(residuals)

# 80% prediction interval
next_day_lower = next_day_ensemble - 1.28 * residual_std
next_day_upper = next_day_ensemble + 1.28 * residual_std

# 95% prediction interval
next_day_lower_95 = next_day_ensemble - 1.96 * residual_std
next_day_upper_95 = next_day_ensemble + 1.96 * residual_std

# -------------------------------------------------
# DROP PROBABILITY PREDICTION
# -------------------------------------------------

# Estimate drop probability based on prediction vs threshold
if "efficiency_threshold_overall" in df_clean.columns:
    threshold = df_clean["efficiency_threshold_overall"].iloc[-1]
    
    # Use prediction interval to estimate probability
    # P(drop) ≈ P(efficiency < threshold)
    # Assume normal distribution of predictions
    z_score = (threshold - next_day_ensemble) / residual_std
    
    from scipy import stats
    drop_probability = stats.norm.cdf(z_score) * 100
else:
    drop_probability = 0
    threshold = 0

# -------------------------------------------------
# SCENARIO ANALYSIS
# -------------------------------------------------

print(f"\n📊 Running scenario analysis...")

scenarios = {
    "Conservative Plan (3 tasks, light workload)": {
        "number_of_tasks": 3,
        "total_planned": 4.0,
        "planning_pressure": 0.8,
        "task_quality_mean": 4.5
    },
    "Normal Plan (5 tasks, moderate workload)": {
        "number_of_tasks": 5,
        "total_planned": 8.0,
        "planning_pressure": 1.0,
        "task_quality_mean": 4.0
    },
    "Ambitious Plan (7 tasks, heavy workload)": {
        "number_of_tasks": 7,
        "total_planned": 12.0,
        "planning_pressure": 1.3,
        "task_quality_mean": 3.5
    }
}

scenario_results = {}

for scenario_name, changes in scenarios.items():
    # Create modified feature set
    scenario_features = latest_features.copy()
    
    for feature, value in changes.items():
        if feature in scenario_features.columns:
            scenario_features[feature] = value
    
    # Update derived features
    if "avg_task_duration" in scenario_features.columns:
        scenario_features["avg_task_duration"] = (
            scenario_features["total_planned"] / scenario_features["number_of_tasks"]
        )
    
    # Scale and predict
    scenario_scaled = scaler.transform(scenario_features)
    scenario_pred = best_model.predict(scenario_scaled)[0]
    
    scenario_results[scenario_name] = scenario_pred

# -------------------------------------------------
# SAVE OUTPUTS
# -------------------------------------------------

# Add predictions to test set
test_results = test_df.copy()
test_results["predicted_efficiency"] = ensemble_pred
test_results["prediction_error"] = test_results["daily_efficiency"] - ensemble_pred

# Save predictions
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
test_results.to_csv(OUTPUT_FILE, index=False)

# Save feature importance
importance_df.to_csv(WEIGHTS_FILE, index=False)

# -------------------------------------------------
# DISPLAY COMPREHENSIVE REPORT
# -------------------------------------------------

print("\n" + "="*80)
print("🤖 MACHINE LEARNING FORECASTING REPORT")
print("="*80)

# Model Performance Comparison
print(f"\n📊 Model Performance (Test Set - Last {TEST_SIZE} Days):")
print("-" * 80)
print(performance_df.round(4).to_string(index=False))
print("-" * 80)

# Ensemble performance
print(f"\n🎯 Ensemble Model Performance:")
print(f"   Test MAE:  {ensemble_mae:.4f}")
print(f"   Test RMSE: {ensemble_rmse:.4f}")
print(f"   Test R²:   {ensemble_r2:.4f}")
print(f"   Test MAPE: {ensemble_mape:.2f}%")

# Model weights
print(f"\n⚖️  Ensemble Weights:")
for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
    print(f"   {name:20s}: {weight:.3f} ({weight*100:.1f}%)")

# Top features
print(f"\n🔑 Top 15 Most Important Features ({best_model_name}):")
print("-" * 80)
top_features = importance_df.head(15)[["feature", "category", "abs_importance"]]
for idx, row in top_features.iterrows():
    print(f"   {row['feature']:35s} ({row['category']:20s}): {row['abs_importance']:.4f}")

# Feature category importance
print(f"\n📂 Feature Importance by Category:")
category_importance = importance_df.groupby("category")["abs_importance"].sum().sort_values(ascending=False)
for category, total_importance in category_importance.items():
    pct = total_importance / category_importance.sum() * 100
    print(f"   {category:25s}: {pct:5.1f}%")

# Next-day predictions
print(f"\n🔮 Tomorrow's Efficiency Predictions:")
print("-" * 80)
for name, pred in sorted(next_day_predictions.items(), key=lambda x: x[1], reverse=True):
    print(f"   {name:20s}: {pred:.3f}")
print("-" * 80)
print(f"   {'ENSEMBLE':20s}: {next_day_ensemble:.3f}")
print(f"\n   80% Prediction Interval: [{next_day_lower:.3f}, {next_day_upper:.3f}]")
print(f"   95% Prediction Interval: [{next_day_lower_95:.3f}, {next_day_upper_95:.3f}]")

# Drop probability
print(f"\n⚠️  Drop Risk Assessment:")
print(f"   Efficiency Threshold: {threshold:.3f}")
print(f"   Predicted Efficiency: {next_day_ensemble:.3f}")
print(f"   Drop Probability: {drop_probability:.1f}%")

if drop_probability >= 60:
    print(f"   Risk Level: 🚨 HIGH - Strong likelihood of efficiency drop")
elif drop_probability >= 30:
    print(f"   Risk Level: ⚡ MODERATE - Elevated risk of underperformance")
else:
    print(f"   Risk Level: ✅ LOW - Normal performance expected")

# Scenario analysis
print(f"\n🎲 Scenario Analysis (What-If Planning):")
print("-" * 80)
for scenario, predicted_eff in sorted(scenario_results.items(), key=lambda x: x[1], reverse=True):
    delta = predicted_eff - next_day_ensemble
    print(f"   {scenario:45s}: {predicted_eff:.3f} ({delta:+.3f})")
print("-" * 80)

# Actionable insights
print(f"\n💡 Key Insights:")

# Identify most impactful features
top_3_features = importance_df.head(3)["feature"].tolist()
print(f"   • Top 3 drivers of efficiency: {', '.join(top_3_features)}")

# Category insights
top_category = category_importance.index[0]
top_category_pct = category_importance.iloc[0] / category_importance.sum() * 100
print(f"   • {top_category.replace('_', ' ').title()} features drive {top_category_pct:.0f}% of predictions")

# Scenario recommendation
best_scenario = max(scenario_results.items(), key=lambda x: x[1])
print(f"   • Optimal planning strategy: {best_scenario[0]} → {best_scenario[1]:.3f} predicted efficiency")

# Model reliability
if ensemble_r2 > 0.7:
    reliability = "High"
elif ensemble_r2 > 0.5:
    reliability = "Moderate"
else:
    reliability = "Low"
print(f"   • Model reliability: {reliability} (R² = {ensemble_r2:.3f})")

print("\n" + "="*80)
print(f"✅ Predictions saved to: {OUTPUT_FILE}")
print(f"✅ Feature weights saved to: {WEIGHTS_FILE}")
print("="*80 + "\n")

# -------------------------------------------------
# OPTIONAL: Plot actual vs predicted (if matplotlib available)
# -------------------------------------------------

try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Actual vs Predicted
    axes[0].plot(range(len(y_test)), y_test.values, 'o-', label='Actual', linewidth=2, markersize=8)
    axes[0].plot(range(len(ensemble_pred)), ensemble_pred, 's--', label='Predicted', linewidth=2, markersize=8)
    axes[0].fill_between(range(len(ensemble_pred)), 
                          ensemble_pred - 1.28 * residual_std,
                          ensemble_pred + 1.28 * residual_std,
                          alpha=0.3, label='80% PI')
    axes[0].set_xlabel('Test Day')
    axes[0].set_ylabel('Efficiency')
    axes[0].set_title(f'Model Performance: {best_model_name} Ensemble (MAE={ensemble_mae:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Feature Importance
    top_10 = importance_df.head(10)
    axes[1].barh(range(len(top_10)), top_10["abs_importance"])
    axes[1].set_yticks(range(len(top_10)))
    axes[1].set_yticklabels(top_10["feature"])
    axes[1].set_xlabel('Absolute Importance')
    axes[1].set_title(f'Top 10 Feature Importance ({best_model_name})')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = OUTPUT_FILE.parent / "model2_performance.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Performance plots saved to: {plot_file}\n")
    
    # Don't show plot in headless environments
    # plt.show()
    
except ImportError:
    print("📊 matplotlib not available - skipping plots\n")