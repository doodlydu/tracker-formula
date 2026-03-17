import math

def set_day_params(day_type: str):
    """
    Returns parameters for a given day type.
    """
    day_type = day_type.lower()
    params = {
    "efficient": {"gamma": 1.20, "delta": 1.00, "alpha": 0.70, "cap": 1.50, "penalty": 0.80, "completion_bonus": 1.08},
    "normal":    {"gamma": 1.00, "delta": 0.80, "alpha": 0.50, "cap": 1.40, "penalty": 0.90, "completion_bonus": 1.04},
    "chill":     {"gamma": 0.60, "delta": 0.50, "alpha": 0.30, "cap": 1.20, "penalty": 1.00, "completion_bonus": 1.00},
}

    if day_type not in params:
        raise ValueError("Invalid day type. Choose from: Efficient, Normal, Chill")

    return params[day_type]


def compute_AR(a: float, pt: float, day_params: dict, T_ref: float = 2.0, k: float = 0.8,quality: float = 1.0,importance: float = 1.0):
    """
    Computes Achievement Ratio (AR).
    """
    if pt <= 0:
        raise ValueError("Planned time must be > 0")

    r = a / pt
    alpha = day_params["alpha"]
    cap = day_params["cap"]
    penalty = day_params["penalty"]

    if r < 1:
        penalty_eff = penalty + (1 - penalty) * (quality * 0.55 + importance * 0.35)
        AR = r * penalty_eff
    else:
        AR = min(1 + alpha * (r - 1), cap)

    return AR


def shape_ar(ar, params,importance=1.0):
    gamma = params["gamma"]
    delta = params["delta"]

    if importance < 0.6:
        scale = 0.5 + (importance / 0.6) * 0.5
        gamma = gamma * scale

    if ar <= 1:
        return ar ** gamma
    else:
        return 1 + delta * (ar - 1)


def compute_task_score(planned_time, achieved_time, importance, quality, params, day_type="None"):
    """
    Compute score for a single task.
    AR dominates, importance/quality moderate,
    and day type has small influence.
    """
    ar = compute_AR(achieved_time, planned_time, params,importance,quality)
    ar_shaped = shape_ar(ar, params,importance  )

    if achieved_time >= planned_time:             # perfect or over completion
        ar_shaped *= params["completion_bonus"]   # Efficient gets most reward

    # Softer importance & quality factor
    iq_factor = 0.3 + 0.7 * (((0.7)*importance +(0.4) *quality) )

    # FIX #2: removed backwards day_factor; shape_ar handles day-type
    # penalisation and reward correctly via gamma and delta
    task_score = ar_shaped * iq_factor
    return task_score


def compute_daily_efficiency(task_scores, importance_list, quality_list, w1=0.5, w2=0.5):
    """
    Compute daily efficiency across tasks.
    """
    numerator = sum(task_scores)
    denominator = sum([w1 * i + w2 * q for i, q in zip(importance_list, quality_list)])
    return numerator / denominator if denominator > 0 else 0