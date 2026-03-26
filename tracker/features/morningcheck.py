# morning_check.py
import pandas as pd

# Quick check: Was yesterday a drop?
df = pd.read_csv("tracker/data/step2_drop_detection.csv")
yesterday = df.iloc[-1]

if yesterday["is_efficiency_drop"]:
    print("🚨 DROP DETECTED - Loading recovery plan...")
    
    # Show recommendations
    rec_df = pd.read_csv("tracker/data/step4_recommendations.csv")
    yesterday_rec = rec_df[rec_df["run_id"] == yesterday["run_id"]].iloc[0]
    
    print("\n" + "═" * 60)
    print(f"Yesterday: {yesterday['daily_efficiency']:.3f} (DROP)")
    print(f"\nPRIMARY CAUSE: {yesterday_rec['primary_cause_name']}")
    print(f"\nACTIONS:")
    print(yesterday_rec["next_steps"])
    print("═" * 60)
else:
    print("✅ No drops detected. Continue with planned tasks!")