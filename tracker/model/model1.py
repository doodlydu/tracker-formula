import pandas as pd

#  Load dataset
df = pd.read_csv("efficiency_dataset.csv")

#  Select efficiency column (last column)
efficiency = df.iloc[:, -1].astype(float)

# Time-series alpha 
N = len(efficiency)
alpha =.4


ema_series = efficiency.ewm(alpha=alpha, adjust=False).mean()


result_df = pd.DataFrame({
    "Day": range(1, N + 1),
    "Efficiency": efficiency.values,
    "EMA_Efficiency": ema_series.round(3)
})


next_day_efficiency = ema_series.iloc[-1]

print("\n--- Past 15 Days Efficiency & EMA ---")
print(result_df.to_string(index=False))

print("\n--- Prediction ---")
print("Alpha used:", round(alpha, 3))
print("Predicted next day efficiency:", round(next_day_efficiency, 3))
