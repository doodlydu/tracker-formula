import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
INPUT_FILE = Path("tracker/data/step3_cause_analysis.csv")
OUTPUT_FILE = Path("tracker/data/step4_recommendations.csv")
HISTORICAL_FILE = Path("tracker/data/step2_drop_detection.csv")

# -------------------------------------------------
# Load Data
# -------------------------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

drop_df = pd.read_csv(INPUT_FILE)

# Load full historical data for pattern analysis
if HISTORICAL_FILE.exists():
    full_df = pd.read_csv(HISTORICAL_FILE)
else:
    full_df = drop_df.copy()

print(f"🔍 Analyzing {len(drop_df)} efficiency drops...")

# -------------------------------------------------
# PATTERN DETECTION ACROSS DROPS
# -------------------------------------------------
def detect_recurring_patterns(df: pd.DataFrame) -> Dict:
    """
    Analyzes historical drops to identify recurring patterns.
    Returns insights about chronic issues.
    """
    patterns = {}
    
    if len(df) < 3:
        return patterns
    
    # Most common primary cause
    if "primary_cause_name" in df.columns:
        cause_counts = df["primary_cause_name"].value_counts()
        if len(cause_counts) > 0:
            most_common = cause_counts.index[0]
            frequency = cause_counts.iloc[0] / len(df)
            patterns["chronic_cause"] = most_common
            patterns["chronic_frequency"] = frequency
    
    # Check for consecutive drops (warning sign)
    if "run_id" in df.columns:
        df_sorted = df.sort_values("run_id")
        consecutive_drops = 0
        max_consecutive = 0
        prev_id = None
        
        for run_id in df_sorted["run_id"]:
            if prev_id is not None and run_id == prev_id + 1:
                consecutive_drops += 1
                max_consecutive = max(max_consecutive, consecutive_drops)
            else:
                consecutive_drops = 1
            prev_id = run_id
        
        patterns["max_consecutive_drops"] = max_consecutive
    
    # Day-type vulnerability (which day types have most drops?)
    if "day_type" in df.columns:
        daytype_counts = df["day_type"].value_counts()
        patterns["vulnerable_daytype"] = daytype_counts.index[0] if len(daytype_counts) > 0 else None
    
    # Average severity
    if "overall_severity" in df.columns:
        patterns["avg_severity"] = df["overall_severity"].mean()
    
    # Multi-cause tendency (complex vs simple problems)
    if "cause_count" in df.columns:
        patterns["avg_causes_per_drop"] = df["cause_count"].mean()
    
    return patterns

patterns = detect_recurring_patterns(drop_df)

# -------------------------------------------------
# RECOMMENDATION ENGINE
# -------------------------------------------------

class RecommendationEngine:
    """
    Intelligent recommendation generator that considers:
    - Primary and secondary causes
    - Severity level
    - Day type context
    - Historical patterns
    - Multi-cause interactions
    """
    
    def __init__(self, patterns: Dict):
        self.patterns = patterns
        
        # Recommendation templates by cause
        self.recommendations = {
            "dev_completion": {
                "mild": self._rec_completion_mild,
                "moderate": self._rec_completion_moderate,
                "severe": self._rec_completion_severe
            },
            "dev_overload": {
                "mild": self._rec_overload_mild,
                "moderate": self._rec_overload_moderate,
                "severe": self._rec_overload_severe
            },
            "dev_planning": {
                "mild": self._rec_planning_mild,
                "moderate": self._rec_planning_moderate,
                "severe": self._rec_planning_severe
            },
            "dev_prioritization": {
                "mild": self._rec_prioritization_mild,
                "moderate": self._rec_prioritization_moderate,
                "severe": self._rec_prioritization_severe
            },
            "dev_quality": {
                "mild": self._rec_quality_mild,
                "moderate": self._rec_quality_moderate,
                "severe": self._rec_quality_severe
            },
            "dev_inconsistency": {
                "mild": self._rec_inconsistency_mild,
                "moderate": self._rec_inconsistency_moderate,
                "severe": self._rec_inconsistency_severe
            },
            "dev_intensity": {
                "mild": self._rec_intensity_mild,
                "moderate": self._rec_intensity_moderate,
                "severe": self._rec_intensity_severe
            },
            "dev_momentum": {
                "mild": self._rec_momentum_mild,
                "moderate": self._rec_momentum_moderate,
                "severe": self._rec_momentum_severe
            },
            "dev_daytype": {
                "mild": self._rec_daytype_mild,
                "moderate": self._rec_daytype_moderate,
                "severe": self._rec_daytype_severe
            }
        }
    
    # -------------------------------------------------
    # COMPLETION RATE RECOMMENDATIONS
    # -------------------------------------------------
    def _rec_completion_mild(self, row: pd.Series) -> str:
        base = "You completed slightly less than usual. "
        
        if row.get("day_type", "").lower() == "efficient":
            return base + "For Efficient days, aim to complete at least 90% of planned tasks. Consider reducing scope by 10% or extending one task to tomorrow."
        elif row.get("day_type", "").lower() == "chill":
            return base + "Even on Chill days, try to hit 75% completion. Break larger tasks into smaller chunks."
        else:
            return base + "Try to maintain 85%+ completion rate. Review if any tasks were unrealistic."
    
    def _rec_completion_moderate(self, row: pd.Series) -> str:
        completion_ratio = row.get("completion_ratio", 0)
        base = f"Completion rate was {completion_ratio*100:.0f}%, below your baseline. "
        
        actions = [
            "Review your task breakdown - were some tasks too large?",
            "Identify which tasks took longer than expected and why",
            "Tomorrow, plan 20% less total work to rebuild momentum"
        ]
        
        return base + " Action steps: " + "; ".join(actions)
    
    def _rec_completion_severe(self, row: pd.Series) -> str:
        completion_ratio = row.get("completion_ratio", 0)
        base = f"⚠️ CRITICAL: Only {completion_ratio*100:.0f}% completion - significantly below normal. "
        
        if self.patterns.get("chronic_frequency", 0) > 0.5:
            base += "This is a recurring pattern. "
        
        actions = [
            "IMMEDIATE: Tomorrow, plan only 3 high-priority tasks (no more)",
            "Conduct a weekly planning review - are you consistently over-ambitious?",
            "Consider time-boxing: allocate specific hours to each task",
            "Track actual vs estimated time for next 5 tasks to calibrate expectations"
        ]
        
        return base + " Recovery plan: " + "; ".join(actions)
    
    # -------------------------------------------------
    # OVERLOAD RECOMMENDATIONS
    # -------------------------------------------------
    def _rec_overload_mild(self, row: pd.Series) -> str:
        task_count = row.get("number_of_tasks", 0)
        baseline = row.get("task_count_mean", 0)
        
        return f"You took on {task_count} tasks vs your usual {baseline:.1f}. Consider reducing to {int(baseline)} tasks tomorrow to recover."
    
    def _rec_overload_moderate(self, row: pd.Series) -> str:
        task_count = row.get("number_of_tasks", 0)
        baseline = row.get("task_count_mean", 0)
        excess = task_count - baseline
        
        base = f"Task overload detected: {excess:.0f} tasks above baseline. "
        
        actions = [
            f"Reduce task count to {int(baseline)-1} for next 2 days",
            "Defer low-importance tasks to next week",
            "Batch similar tasks together to reduce context-switching overhead"
        ]
        
        return base + " Actions: " + "; ".join(actions)
    
    def _rec_overload_severe(self, row: pd.Series) -> str:
        task_count = row.get("number_of_tasks", 0)
        baseline = row.get("task_count_mean", 0)
        
        base = f"⚠️ SEVERE OVERLOAD: {task_count} tasks (baseline: {baseline:.1f}). "
        
        if self.patterns.get("chronic_cause") == "Task Overload":
            base += "This is your most common drop cause - you need a systematic fix. "
        
        actions = [
            f"IMMEDIATE: Cancel/defer at least {int(task_count - baseline + 2)} tasks",
            "Implement a daily task limit: MAX {int(baseline)} tasks per day",
            "Use the 3-task rule: Start each day with only 3 must-do tasks",
            "Review your task acceptance criteria - learn to say NO"
        ]
        
        return base + " Critical actions: " + "; ".join(actions)
    
    # -------------------------------------------------
    # PLANNING PRESSURE RECOMMENDATIONS
    # -------------------------------------------------
    def _rec_planning_mild(self, row: pd.Series) -> str:
        pressure = row.get("planning_pressure", 1.0)
        return f"Planning pressure at {pressure:.1f}x (planned/achieved ratio). Reduce planned time by 10% tomorrow."
    
    def _rec_planning_moderate(self, row: pd.Series) -> str:
        pressure = row.get("planning_pressure", 1.0)
        base = f"Over-ambitious planning: {pressure:.1f}x ratio. "
        
        actions = [
            "Add 25% buffer time to all estimates",
            "Review which tasks took longer - update your estimation model",
            "Use historical data: check how long similar tasks actually took"
        ]
        
        return base + " Fixes: " + "; ".join(actions)
    
    def _rec_planning_severe(self, row: pd.Series) -> str:
        pressure = row.get("planning_pressure", 1.0)
        base = f"⚠️ UNREALISTIC PLANNING: {pressure:.1f}x over-estimation. "
        
        if row.get("day_type", "").lower() == "efficient":
            base += "Even Efficient days need realistic targets. "
        
        actions = [
            "RESET: Tomorrow, plan only 50% of what you think you can do",
            "Track time for 1 week to learn your true capacity",
            "Use the planning fallacy correction: double your initial estimate",
            "Consider Parkinson's Law: work expands to fill time - plan LESS"
        ]
        
        return base + " Recalibration needed: " + "; ".join(actions)
    
    # -------------------------------------------------
    # PRIORITIZATION RECOMMENDATIONS
    # -------------------------------------------------
    def _rec_prioritization_mild(self, row: pd.Series) -> str:
        return "Slightly more effort went to lower-priority tasks. Tomorrow, tackle your highest-importance task first (before checking email/messages)."
    
    def _rec_prioritization_moderate(self, row: pd.Series) -> str:
        base = "Prioritization issues: too much effort on low-importance work. "
        
        actions = [
            "Use Eisenhower Matrix: Focus on Important+Urgent first",
            "Implement the 'eat the frog' rule: Hardest/most important task at start of day",
            "Time-block your top 3 priorities before allowing interruptions"
        ]
        
        return base + " Strategy: " + "; ".join(actions)
    
    def _rec_prioritization_severe(self, row: pd.Series) -> str:
        importance_ratio = row.get("importance_effort_ratio", 0)
        base = f"⚠️ CRITICAL PRIORITY FAILURE: Low-importance work dominated your day. "
        
        actions = [
            "AUDIT: List every task you did - honestly rate importance (1-5)",
            "Implement a strict rule: No tasks below importance=3 until all 4-5 tasks done",
            "Identify time thieves: What low-value activities consumed your time?",
            "Create a 'NOT-TO-DO' list: Activities you will actively avoid",
            "Consider: Are you procrastinating on hard/important work?"
        ]
        
        return base + " Deep fix needed: " + "; ".join(actions)
    
    # -------------------------------------------------
    # QUALITY DROP RECOMMENDATIONS
    # -------------------------------------------------
    def _rec_quality_mild(self, row: pd.Series) -> str:
        return "Quality slightly below baseline. Ensure you're taking breaks and minimizing distractions on important work."
    
    def _rec_quality_moderate(self, row: pd.Series) -> str:
        quality_mean = row.get("task_quality_mean", 0)
        base = f"Quality drop detected (avg quality: {quality_mean:.2f}). "
        
        actions = [
            "Reduce multitasking: Single-task on important work",
            "Use Pomodoro technique: 25min focused work + 5min break",
            "Identify what distracted you today - block those interruptions tomorrow"
        ]
        
        return base + " Focus improvements: " + "; ".join(actions)
    
    def _rec_quality_severe(self, row: pd.Series) -> str:
        quality_mean = row.get("task_quality_mean", 0)
        base = f"⚠️ QUALITY CRISIS: Average quality {quality_mean:.2f} - well below your standard. "
        
        actions = [
            "STOP: Are you burned out? Consider taking a recovery day",
            "Environment audit: Remove ALL distractions from workspace",
            "Deep work blocks: 90-min sessions with phone off, notifications blocked",
            "Energy management: Are you working during your low-energy hours?",
            "Review sleep/health: Poor quality often signals fatigue or stress"
        ]
        
        return base + " Quality recovery: " + "; ".join(actions)
    
    # -------------------------------------------------
    # INCONSISTENCY RECOMMENDATIONS
    # -------------------------------------------------
    def _rec_inconsistency_mild(self, row: pd.Series) -> str:
        return "Task performance was uneven. Review what went well vs poorly - replicate success patterns."
    
    def _rec_inconsistency_moderate(self, row: pd.Series) -> str:
        base = "High task score variance: Some tasks excellent, others poor. "
        
        actions = [
            "Analyze: Which tasks scored high vs low? Find patterns",
            "Time-of-day effect: Schedule hard tasks during peak energy hours",
            "Task similarity: Group similar tasks to maintain flow state"
        ]
        
        return base + " Consistency fixes: " + "; ".join(actions)
    
    def _rec_inconsistency_severe(self, row: pd.Series) -> str:
        base = "⚠️ EXTREME INCONSISTENCY: Performance wildly varied across tasks. "
        
        actions = [
            "Root cause analysis: What made good tasks succeed and bad tasks fail?",
            "Context switching cost: Too many different task types in one day",
            "Skill gaps: Did some tasks expose areas needing more knowledge/practice?",
            "Energy allocation: Were you depleted by time you hit certain tasks?",
            "Tomorrow: Reduce task variety - stick to 2-3 similar task types max"
        ]
        
        return base + " Deep analysis needed: " + "; ".join(actions)
    
    # -------------------------------------------------
    # INTENSITY RECOMMENDATIONS
    # -------------------------------------------------
    def _rec_intensity_mild(self, row: pd.Series) -> str:
        total_planned = row.get("total_planned", 0)
        return f"Workload intensity slightly high ({total_planned:.1f} hours planned). Reduce by 1-2 hours tomorrow."
    
    def _rec_intensity_moderate(self, row: pd.Series) -> str:
        total_planned = row.get("total_planned", 0)
        baseline = row.get("workload_intensity_mean", 0)
        
        base = f"High intensity: {total_planned:.1f} hours vs baseline {baseline:.1f}. "
        
        actions = [
            f"Reduce to {baseline-1:.0f} hours for next 2 days",
            "Add buffer time between tasks for cognitive recovery",
            "Consider: Are you underestimating task complexity?"
        ]
        
        return base + " Workload adjustments: " + "; ".join(actions)
    
    def _rec_intensity_severe(self, row: pd.Series) -> str:
        total_planned = row.get("total_planned", 0)
        baseline = row.get("workload_intensity_mean", 0)
        
        base = f"⚠️ UNSUSTAINABLE INTENSITY: {total_planned:.1f} hours (baseline: {baseline:.1f}). "
        
        actions = [
            f"IMMEDIATE: Cut planned hours to {baseline*0.7:.0f} tomorrow",
            "This is burnout territory - take a half-day recovery break",
            "Implement sustainable pace: 4-6 focused hours/day MAX",
            "Review: Are external pressures forcing this pace? Address root cause",
            "Long-term: Build in 20% slack time for sustainability"
        ]
        
        return base + " Urgent intervention: " + "; ".join(actions)
    
    # -------------------------------------------------
    # MOMENTUM RECOMMENDATIONS
    # -------------------------------------------------
    def _rec_momentum_mild(self, row: pd.Series) -> str:
        return "Slight downward trend detected. Focus on one 'win' tomorrow to reverse momentum."
    
    def _rec_momentum_moderate(self, row: pd.Series) -> str:
        days_declining = row.get("days_since_improvement", 0)
        base = f"Declining trend: {days_declining} days since improvement. "
        
        actions = [
            "Break the pattern: Set one easy, achievable goal for tomorrow",
            "Review what changed: Compare to your last good day",
            "Small win strategy: Complete 3 quick tasks to rebuild confidence"
        ]
        
        return base + " Momentum recovery: " + "; ".join(actions)
    
    def _rec_momentum_severe(self, row: pd.Series) -> str:
        days_declining = row.get("days_since_improvement", 0)
        trend = row.get("eff_trend_medium", 0)
        
        base = f"⚠️ EXTENDED DECLINE: {days_declining} days without improvement (trend: {trend:.3f}). "
        
        actions = [
            "RESET NEEDED: Take a planning day to reassess your system",
            "What's changed in your life/work causing this slide?",
            "Simplify everything: Go back to basics for 3 days",
            "Set ridiculously easy targets to rebuild momentum and confidence",
            "Consider: Do you need external help/support to break this pattern?",
            "Track daily: What small thing can you improve each day?"
        ]
        
        return base + " Pattern interrupt required: " + "; ".join(actions)
    
    # -------------------------------------------------
    # DAY-TYPE UNDERPERFORMANCE RECOMMENDATIONS
    # -------------------------------------------------
    def _rec_daytype_mild(self, row: pd.Series) -> str:
        day_type = row.get("day_type", "").title()
        return f"Performance below average for your {day_type} days. Review what made your best {day_type} days successful."
    
    def _rec_daytype_moderate(self, row: pd.Series) -> str:
        day_type = row.get("day_type", "").title()
        percentile = row.get("daytype_percentile", 50)
        
        base = f"This {day_type} day ranked in bottom {100-percentile:.0f}% of your {day_type} days. "
        
        if day_type.lower() == "efficient":
            actions = [
                "Efficient days need: High focus, minimal distractions, important work",
                "Did you treat this like a Normal day? Reset expectations",
                "Review your best Efficient days - what conditions enabled them?"
            ]
        elif day_type.lower() == "chill":
            actions = [
                "Even Chill days need structure - just lower intensity",
                "Did you over-plan? Chill days = fewer, easier tasks",
                "Chill doesn't mean unproductive - just sustainable pace"
            ]
        else:  # Normal
            actions = [
                "Normal days are your baseline - this should be sustainable",
                "What disrupted today? Identify and prevent for next Normal day",
                "Consistency matters: Normal days should feel... normal"
            ]
        
        return base + " Day-type alignment: " + "; ".join(actions)
    
    def _rec_daytype_severe(self, row: pd.Series) -> str:
        day_type = row.get("day_type", "").title()
        delta = row.get("daytype_performance_delta", 0)
        
        base = f"⚠️ MISMATCHED DAY TYPE: This {day_type} day performed far below your {day_type} baseline. "
        
        if day_type.lower() == "efficient":
            guidance = "EFFICIENT DAY FAILURE: Either you mis-labeled the day (should have been Normal/Chill) OR you attempted Efficient-level goals without Efficient-level focus. Tomorrow: If planning Efficient, ensure environment, energy, and task list all support high performance. Otherwise, plan a Normal day."
        elif day_type.lower() == "chill":
            guidance = "CHILL DAY FAILURE: You may have over-planned or tried to force productivity. Chill days need: fewer tasks, lower stakes, more flexibility. Don't judge Chill days by Efficient standards."
        else:
            guidance = "NORMAL DAY FAILURE: Your baseline day type should be reliable. This suggests either burnout, external factors, or miscalibrated expectations. Review your 'Normal' definition - is it actually sustainable?"
        
        return base + guidance
    
    # -------------------------------------------------
    # MAIN RECOMMENDATION GENERATOR
    # -------------------------------------------------
    def generate_recommendation(self, row: pd.Series) -> Tuple[str, str, str]:
        """
        Generates primary, secondary, and systemic recommendations.
        
        Returns:
            (primary_rec, secondary_rec, systemic_rec)
        """
        primary_cause = row.get("primary_cause", "")
        secondary_cause = row.get("secondary_cause", "")
        severity = row.get("severity_level", "mild")
        
        # Primary recommendation
        primary_rec = "Review task planning and execution."
        if primary_cause in self.recommendations:
            handler = self.recommendations[primary_cause].get(severity)
            if handler:
                primary_rec = handler(row)
        
        # Secondary recommendation
        secondary_rec = ""
        if secondary_cause and secondary_cause in self.recommendations:
            handler = self.recommendations[secondary_cause].get("mild")  # Always mild for secondary
            if handler:
                secondary_rec = handler(row)
        
        # Systemic/pattern-based recommendation
        systemic_rec = self._generate_systemic_recommendation(row)
        
        return primary_rec, secondary_rec, systemic_rec
    
    def _generate_systemic_recommendation(self, row: pd.Series) -> str:
        """
        Generates recommendations based on patterns across multiple drops.
        """
        recs = []
        
        # Chronic issue
        if self.patterns.get("chronic_frequency", 0) > 0.5:
            chronic = self.patterns.get("chronic_cause", "")
            recs.append(f"⚠️ PATTERN ALERT: '{chronic}' caused {self.patterns['chronic_frequency']*100:.0f}% of your drops. This needs systematic intervention, not just daily fixes.")
        
        # Consecutive drops
        if self.patterns.get("max_consecutive_drops", 0) >= 3:
            recs.append(f"🔴 WARNING: You've had {self.patterns['max_consecutive_drops']} consecutive drop days. Consider taking a reset day to break the cycle.")
        
        # Multi-cause complexity
        if row.get("cause_count", 0) >= 3:
            recs.append("🔍 COMPLEX ISSUE: Multiple factors contributed. Focus on the top 2 causes first - don't try to fix everything at once.")
        
        # High severity
        if row.get("overall_severity", 0) > 2.5:
            recs.append("🚨 SEVERE DROP: This requires immediate intervention. Consider whether you need external support or a larger life/work adjustment.")
        
        # Day-type vulnerability
        vulnerable = self.patterns.get("vulnerable_daytype")
        if vulnerable and row.get("day_type", "").lower() == vulnerable.lower():
            recs.append(f"📊 INSIGHT: {vulnerable} days are your most vulnerable. Consider whether this day-type label is appropriate or needs recalibration.")
        
        return " | ".join(recs) if recs else "Continue monitoring patterns."

# -------------------------------------------------
# Generate Recommendations for All Drops
# -------------------------------------------------
engine = RecommendationEngine(patterns)

recommendations_list = []
for idx, row in drop_df.iterrows():
    primary_rec, secondary_rec, systemic_rec = engine.generate_recommendation(row)
    recommendations_list.append({
        "primary_recommendation": primary_rec,
        "secondary_recommendation": secondary_rec,
        "systemic_recommendation": systemic_rec,
        "combined_recommendation": f"PRIMARY: {primary_rec}\n\nSECONDARY: {secondary_rec}\n\nSYSTEMIC: {systemic_rec}"
    })

rec_df = pd.DataFrame(recommendations_list)
drop_df = pd.concat([drop_df.reset_index(drop=True), rec_df], axis=1)

# -------------------------------------------------
# Add Actionable Next Steps
# -------------------------------------------------
def generate_next_steps(row: pd.Series) -> str:
    """Quick, actionable steps for tomorrow."""
    steps = []
    
    severity = row.get("severity_level", "mild")
    primary = row.get("primary_cause", "")
    
    # Always: Adjust tomorrow's plan
    if "completion" in primary:
        steps.append("[ ] Reduce tomorrow's task count by 20-30%")
    elif "overload" in primary:
        steps.append("[ ] Limit tomorrow to 3-5 tasks MAX")
    elif "planning" in primary:
        steps.append("[ ] Add 50% buffer time to all estimates")
    elif "prioritization" in primary:
        steps.append("[ ] Start with your #1 most important task")
    elif "quality" in primary:
        steps.append("[ ] Block 2-hour distraction-free focus time")
    
    # Severity-based
    if severity == "severe":
        steps.append("[ ] Consider: Do you need a recovery/reset day?")
        steps.append("[ ] Review: What external factors contributed?")
    
    # Pattern-based
    if patterns.get("chronic_frequency", 0) > 0.5:
        steps.append("[ ] Schedule 30min to address your recurring pattern")
    
    return "\n".join(steps) if steps else "[ ] Review and adjust tomorrow's plan"

drop_df["next_steps"] = drop_df.apply(generate_next_steps, axis=1)

# -------------------------------------------------
# Save Enhanced Recommendations
# -------------------------------------------------
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
drop_df.to_csv(OUTPUT_FILE, index=False)

print("✅ Step 4 completed. Enhanced recommendations generated.")
print(f"\n📋 Recommendation Summary:")
print(f"  • Total drops analyzed: {len(drop_df)}")
print(f"  • Severity breakdown: {drop_df['severity_level'].value_counts().to_dict()}")

if patterns:
    print(f"\n🔍 Pattern Analysis:")
    if "chronic_cause" in patterns:
        print(f"  • Most common cause: {patterns['chronic_cause']} ({patterns['chronic_frequency']*100:.0f}% of drops)")
    if "max_consecutive_drops" in patterns:
        print(f"  • Max consecutive drops: {patterns['max_consecutive_drops']}")
    if "vulnerable_daytype" in patterns:
        print(f"  • Most vulnerable day type: {patterns['vulnerable_daytype']}")
    if "avg_causes_per_drop" in patterns:
        print(f"  • Avg causes per drop: {patterns['avg_causes_per_drop']:.1f}")

# Show example recommendation
if len(drop_df) > 0:
    print(f"\n📄 Example Recommendation (Day {drop_df.iloc[0].get('run_id', 'N/A')}):")
    print("="*80)
    example = drop_df.iloc[0]
    print(f"\n{example['combined_recommendation']}")
    print(f"\nNEXT STEPS:\n{example['next_steps']}")
    print("="*80)
