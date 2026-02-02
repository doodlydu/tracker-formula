import pandas as pd
from pathlib import Path
df = pd.read_csv("tracker/data/step3_cause_analysis.csv")

# -------------------------------------------------
# Recommendation generator
# -------------------------------------------------
def generate_recommendation(row):
    cause = row["primary_cause"]

    severity = abs(row[cause])

    if cause == "dev_overload":
        if severity > 0.6:
            return "Workload was significantly higher than usual. Reduce the number of tasks planned."
        else:
            return "Workload was slightly above normal. Consider planning one less task."

    if cause == "dev_completion":
        return "Planned work exceeded completion capacity. Reduce scope or break tasks into smaller units."

    if cause == "dev_planning":
        return "Targets were overly ambitious. Consider lowering planned time or setting more achievable goals."

    if cause == "dev_prioritization":
        return "More effort went to low-priority tasks. Focus on completing high-importance tasks first."

    if cause == "dev_quality":
        return "Quality of important work dropped. Reduce distractions or allocate focused time blocks."

    return "Review task planning and execution for this day."

# -------------------------------------------------
# Apply recommendations
# -------------------------------------------------
df["recommendation"] = df.apply(generate_recommendation, axis=1)
# -------------------------------------------------
# Save Step-4 output
# -------------------------------------------------
df.to_csv("tracker/data/step4_recommendations.csv", index=False)
print("✅ Step 4 completed. Recommendations generated.")
