import math
def set_day_params(day_type: str):
    """
    Returns parameters for a given day type.
    
    Parameters
    ----------
    day_type : str
        One of ["Efficient", "Normal", "Chill"]
    
    Returns
    -------
    dict
        Dictionary with keys: gamma, delta, alpha, cap
    """
    day_type = day_type.lower()
    params = {
        "efficient": {"gamma": 1.20, "delta": 1.00, "alpha": 0.70, "cap": 1.50},
        "normal":    {"gamma": 1.00, "delta": 0.80, "alpha": 0.50, "cap": 1.40},
        "chill":     {"gamma": 0.60, "delta": 0.50, "alpha": 0.30, "cap": 1.20}
    }
    
    if day_type not in params:
        raise ValueError("Invalid day type. Choose from: Efficient, Normal, Chill")
    
    return params[day_type]

def compute_AR(a: float, pt: float, day_params: dict, T_ref: float = 2.0, k: float = 0.8):
    """
    Computes Achievement Ratio (AR).
    
    Parameters
    ----------
    a : float
        Actual hours spent
    pt : float
        Planned target hours
    day_params : dict
        Parameters from set_day_params (must include alpha, cap)
    T_ref : float
        Typical task length (default=2h)
    k : float
        Penalty factor for small pt (0 <= k <= 1)
    
    Returns
    -------
    float
        Achievement Ratio (AR)
    """
    if pt <= 0:
        raise ValueError("Planned time must be > 0")

    r = a / pt  
    alpha = day_params["alpha"]
    cap = day_params["cap"]
    
    if r < 1:
        beta = 1 + k * (T_ref / (T_ref + pt))
        AR = r * beta
    else:
        AR = min(1 + alpha * (r - 1), cap)
    
    return AR

def shape_ar(ar, params):
    gamma = 1 + (params["gamma"] - 1) * 0.5   # reduce penalty
    delta = 1 + (params["delta"] - 1) * 0.5   # reduce generosity
    
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
    ar = achieved_time / planned_time if planned_time > 0 else 0


    # Softer AR shaping (dominant factor)
    if ar<0.5:
        ar_shaped=0.2+0.4*ar
    elif ar <= 1:
        ar_shaped = 0.35 + 0.6 * ar
    else:
        ar_shaped = 1 + 0.8 * (ar - 1)

    # Softer importance & quality factor
    iq_factor = 0.3 + 0.7 * (((0.7)*importance +(0.4) *quality) )

    # Small effect of day type
    day_factor = 1.5 - 0.4 * params.get("cap", 1)       

    # Task score
    task_score = ar_shaped * iq_factor * day_factor
    return task_score


def compute_daily_efficiency(task_scores, importance_list, quality_list, w1=0.5, w2=0.5):
    """
    Compute daily efficiency across tasks.
    task_scores: list of task scores (from compute_task_score)
    importance_list, quality_list: lists of task importance and quality
    """
    numerator = sum(task_scores)
    denominator = sum([w1 * i + w2 * q for i, q in zip(importance_list, quality_list)])
    return numerator / denominator if denominator > 0 else 0


