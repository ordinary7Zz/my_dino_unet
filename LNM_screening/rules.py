from pathlib import Path

from LNM_screening.features import compute_screen_score
from LNM_screening.io_utils import load_json, load_yaml


DEFAULT_THRESHOLDS = {
    "pred_threshold": 0.5,
    "min_region_area": 32,
    "thresholds": {
        "trust_fg_prob_mean_min": 0.70,
        "trust_prob_max_min": 0.90,
        "trust_largest_component_area_min": 256,
        "trust_positive_fraction_min": 0.002,
        "trust_positive_fraction_max": 0.40,
        "trust_largest_component_ratio_min": 0.60,
        "trust_num_components_max": 3,
        "trust_high_conf_fraction_0p9_min": 0.0005,
    },
    "negative_rules": {
        "max_prob_max_for_no": 0.35,
        "max_fg_prob_mean_for_no": 0.40,
        "max_positive_fraction_for_empty": 0.0005,
    },
    "label_thresholds": {
        "yes_score_min": 0.72,
        "suspicious_score_min": 0.35,
    },
}


def _resolve_default_threshold_path():
    return Path(__file__).with_name("default_thresholds.json")


def _merge_thresholds(data):
    merged = dict(DEFAULT_THRESHOLDS)
    merged.update(data)
    merged["thresholds"] = {**DEFAULT_THRESHOLDS["thresholds"], **data.get("thresholds", {})}
    merged["negative_rules"] = {**DEFAULT_THRESHOLDS["negative_rules"], **data.get("negative_rules", {})}
    merged["label_thresholds"] = {**DEFAULT_THRESHOLDS["label_thresholds"], **data.get("label_thresholds", {})}
    return merged


def load_thresholds(path=None):
    if path is None:
        path = _resolve_default_threshold_path()
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = load_yaml(path)
    else:
        data = load_json(path)
    return _merge_thresholds(data)


def evaluate_trust_rules(features, thresholds):
    trust = thresholds["thresholds"]
    passed = []
    failed = []
    borderline = []

    checks = {
        "trust_fg_prob_mean_min": features.get("fg_prob_mean", 0.0) >= trust["trust_fg_prob_mean_min"],
        "trust_prob_max_min": features.get("prob_max", 0.0) >= trust["trust_prob_max_min"],
        "trust_largest_component_area_min": features.get("largest_component_area", 0.0) >= trust["trust_largest_component_area_min"],
        "trust_positive_fraction_min": features.get("pred_positive_fraction", 0.0) >= trust["trust_positive_fraction_min"],
        "trust_positive_fraction_max": features.get("pred_positive_fraction", 0.0) <= trust["trust_positive_fraction_max"],
        "trust_largest_component_ratio_min": features.get("largest_component_ratio", 0.0) >= trust["trust_largest_component_ratio_min"],
        "trust_num_components_max": features.get("num_components", 0) <= trust["trust_num_components_max"],
        "trust_high_conf_fraction_0p9_min": features.get("high_conf_fraction_0p9", 0.0) >= trust["trust_high_conf_fraction_0p9_min"],
    }

    for name, ok in checks.items():
        if ok:
            passed.append(name)
        else:
            failed.append(name)

    for name, threshold in trust.items():
        value = None
        if name == "trust_fg_prob_mean_min":
            value = features.get("fg_prob_mean", 0.0)
        elif name == "trust_prob_max_min":
            value = features.get("prob_max", 0.0)
        elif name == "trust_largest_component_area_min":
            value = features.get("largest_component_area", 0.0)
        elif name == "trust_positive_fraction_min":
            value = features.get("pred_positive_fraction", 0.0)
        elif name == "trust_positive_fraction_max":
            value = features.get("pred_positive_fraction", 0.0)
        elif name == "trust_largest_component_ratio_min":
            value = features.get("largest_component_ratio", 0.0)
        elif name == "trust_num_components_max":
            value = features.get("num_components", 0)
        elif name == "trust_high_conf_fraction_0p9_min":
            value = features.get("high_conf_fraction_0p9", 0.0)
        if value is None:
            continue
        if "_max" in name:
            margin = threshold * 0.1 if threshold else 1.0
            if threshold < value <= threshold + margin:
                borderline.append(name)
        else:
            margin = threshold * 0.1 if threshold else 0.05
            if threshold - margin <= value < threshold:
                borderline.append(name)

    trustworthy = len(failed) == 0
    return {
        "trustworthy_mask": trustworthy,
        "passed_rules": passed,
        "failed_rules": failed,
        "borderline_rules": borderline,
    }


def assign_screening_label(features, thresholds=None, suspicious_policy="default"):
    thresholds = thresholds or load_thresholds()
    trust_eval = evaluate_trust_rules(features, thresholds)
    score_eval = compute_screen_score(features)
    negative = thresholds["negative_rules"]
    label_thresholds = thresholds["label_thresholds"]

    reasons = []
    hard_reject_flag = False
    if features.get("pred_has_positive", 0) == 0:
        label = "no"
        hard_reject_flag = True
        reasons.append("no_positive_region")
    elif features.get("largest_component_area", 0) < thresholds.get("min_region_area", 32):
        label = "no"
        hard_reject_flag = True
        reasons.append("largest_component_too_small")
    elif (
        features.get("prob_max", 0.0) <= negative["max_prob_max_for_no"]
        and features.get("fg_prob_mean", 0.0) <= negative["max_fg_prob_mean_for_no"]
    ):
        label = "no"
        hard_reject_flag = True
        reasons.append("weak_probability_response")
    elif features.get("pred_positive_fraction", 0.0) <= negative["max_positive_fraction_for_empty"]:
        label = "no"
        hard_reject_flag = True
        reasons.append("nearly_empty_mask")
    elif trust_eval["trustworthy_mask"] and score_eval["screen_score"] >= label_thresholds["yes_score_min"]:
        label = "yes"
        reasons.append("passes_all_trust_rules")
    elif score_eval["screen_score"] >= label_thresholds["suspicious_score_min"]:
        label = "suspicious"
        reasons.append("partial_positive_evidence")
    else:
        label = "no"
        reasons.append("insufficient_positive_evidence")

    if label == "yes" and trust_eval["borderline_rules"]:
        confidence = "medium"
    elif label == "yes":
        confidence = "high"
    elif label == "suspicious":
        confidence = "medium"
    else:
        confidence = "low"

    if trust_eval["failed_rules"]:
        reasons.extend([f"failed:{name}" for name in trust_eval["failed_rules"]])
    if trust_eval["borderline_rules"]:
        reasons.extend([f"borderline:{name}" for name in trust_eval["borderline_rules"]])

    return {
        **trust_eval,
        **score_eval,
        "screening_label": label,
        "confidence": confidence,
        "hard_reject_flag": hard_reject_flag,
        "reason": ";".join(reasons),
    }
