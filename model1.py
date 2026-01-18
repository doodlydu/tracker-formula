import pandas as pd

# 1️⃣ Load dataset
df = pd.read_csv("efficiency_dataset.csv")

# 2️⃣ Select efficiency column (last column)
efficiency = df.iloc[:, -1].astype(float)

# 3️⃣ Time-series alpha (uses ALL 15 days)
N = len(efficiency)
alpha = 4 / (N + 1)

# 4️⃣ Train EMA model
ema_series = efficiency.ewm(alpha=alpha, adjust=False).mean()

# 5️⃣ Create display table
result_df = pd.DataFrame({
    "Day": range(1, N + 1),
    "Efficiency": efficiency.values,
    "EMA_Efficiency": ema_series.round(3)
})

# 6️⃣ Predict next day efficiency
next_day_efficiency = ema_series.iloc[-1]

# 7️⃣ Print results
print("\n--- Past 15 Days Efficiency & EMA ---")
print(result_df.to_string(index=False))

print("\n--- Prediction ---")
print("Alpha used:", round(alpha, 3))
print("Predicted next day efficiency:", round(next_day_efficiency, 3))
