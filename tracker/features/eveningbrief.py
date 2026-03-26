# evening_briefing.py
import subprocess
import pandas as pd
from pathlib import Path

print("═" * 60)
print("📊 EVENING PLANNING REPORT")
print("═" * 60)

# 1. Show today's logged efficiency (if already logged)
latest_data = pd.read_csv("efficiency_dataset.csv")
if len(latest_data) > 0:
    today = latest_data.iloc[-1]
    print(f"\n1️⃣  TODAY'S RESULTS:")
    print(f"   Efficiency: {today['daily_efficiency']:.3f}")
    print(f"   Tasks: {today['number_of_tasks']}")

# 2. Run Model 2 for tomorrow's forecast
print(f"\n2️⃣  RUNNING FORECAST...")
subprocess.run(["python", "model2.py"], capture_output=True)

# Extract and display key predictions
predictions = pd.read_csv("tracker/data/model2_predictions.csv")
# ... extract key metrics and display

print("\n" + "═" * 60)
print("✅ Review forecast above and plan tomorrow accordingly")
print("═" * 60)