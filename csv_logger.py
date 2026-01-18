import csv
import os
from datetime import date

# ✅ DEFINE FIRST
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "efficiency_dataset.csv")
print("CSV created at:", os.path.abspath(CSV_FILE))


MAX_TASKS = 10

HEADERS = (
    ["run_id", "day_type", "number_of_tasks"] +
    [
        f"task_{i}_{field}"
        for i in range(1, MAX_TASKS + 1)
        for field in ["planned", "achieved", "importance", "quality", "score"]
    ] +
    ["daily_efficiency"]
)

def get_next_run_id():
    if not os.path.exists(CSV_FILE):
        return 1

    with open(CSV_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Subtract 1 for header
    return max(1, len(lines))


def log_day(day_type, tasks, daily_efficiency):
    print(">>> log_day() CALLED")
    print(">>> CSV PATH:", CSV_FILE)
    write_headers = not os.path.exists(CSV_FILE)

    run_id = get_next_run_id()

    row = [
        run_id,
        day_type,
        len(tasks)
    ]

    for t in tasks:
        row.extend([
            t["planned"],
            t["achieved"],
            t["importance"],
            t["quality"],
            round(t["score"], 2)
        ])

    for _ in range(MAX_TASKS - len(tasks)):
        row.extend(["", "", "", "", ""])

    row.append(round(daily_efficiency, 2))

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_headers:
            writer.writerow(HEADERS)
        writer.writerow(row)
