from efficiency_core import compute_task_score, set_day_params
from csv_logger import log_day
def compute_daily_efficiency():
    """
    Asks user for inputs, runs task scoring loop, and computes daily efficiency.
    """
    # Ask user for the type of day
    day_type = input("Enter day type (Efficient / Normal / Chill): ").strip()
    params = set_day_params(day_type)

    # Ask how many tasks the user wants to enter
    num_tasks = int(input("How many tasks do you want to input? "))

    weighted_score_sum = 0
    weight_sum = 0
    tasks = []   # NEW: store task data for CSV


    for i in range(num_tasks):
        print(f"\n--- Task {i+1} ---")
        planned_time = float(input("Enter planned time (in hours): "))
        achieved_time = float(input("Enter achieved time (in hours): "))
        importance = float(input("Enter importance (0–1): "))
        quality = float(input("Enter quality (0–1): "))

        task_score = compute_task_score(
            planned_time,
            achieved_time,
            importance,
            quality,
            params
        )
        tasks.append({
    "planned": planned_time,
    "achieved": achieved_time,
    "importance": importance,
    "quality": quality,
    "score": task_score
    })


        print(f"Task {i+1} score = {task_score:.2f}")

        # Weight by importance (so important tasks matter more)
        weighted_score_sum += task_score * importance
        weight_sum += importance

    # Daily efficiency = weighted average of task scores
    daily_efficiency = weighted_score_sum / weight_sum if weight_sum > 0 else 0
    print(f"\n✅ Daily Efficiency = {daily_efficiency:.2f}")

    log_day(day_type, tasks, daily_efficiency)


    return daily_efficiency


# Run when file is executed directly
if __name__ == "__main__":
    compute_daily_efficiency()