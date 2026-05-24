import math
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError


EPS = 1e-8


def sigmoid_logits_to_prob(pred_logits):
    arr = np.asarray(pred_logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-arr))


def prob_to_binary(prob, threshold):
    return (np.asarray(prob, dtype=np.float32) >= float(threshold)).astype(np.uint8)


def _safe_percentile(arr, q):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def _component_slices(labeled, num_components):
    if num_components == 0:
        return []
    return ndimage.find_objects(labeled)


def connected_components_from_binary(binary_mask):
    mask = (np.asarray(binary_mask) > 0).astype(np.uint8)
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, num_components = ndimage.label(mask, structure=structure)
    slices = _component_slices(labeled, num_components)

    components = []
    for idx, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        component_mask = labeled[slc] == idx
        area = int(component_mask.sum())
        if area <= 0:
            continue
        ys, xs = np.where(component_mask)
        y0 = slc[0].start + int(ys.min())
        y1 = slc[0].start + int(ys.max())
        x0 = slc[1].start + int(xs.min())
        x1 = slc[1].start + int(xs.max())
        components.append(
            {
                "label": idx,
                "slice": slc,
                "area": area,
                "bbox": (x0, y0, x1, y1),
                "mask": (labeled == idx),
            }
        )
    components.sort(key=lambda x: x["area"], reverse=True)
    return labeled, components


def _perimeter(binary_mask):
    mask = (np.asarray(binary_mask) > 0)
    if mask.sum() == 0:
        return 0.0
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    boundary = mask ^ eroded
    return float(boundary.sum())


def _convex_hull_area(binary_mask):
    ys, xs = np.where(np.asarray(binary_mask) > 0)
    if ys.size < 3:
        return float(ys.size)
    points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    try:
        hull = ConvexHull(points)
    except QhullError:
        return float(points.shape[0])
    return float(max(hull.volume, points.shape[0]))


def _principal_axes(binary_mask):
    ys, xs = np.where(np.asarray(binary_mask) > 0)
    if ys.size < 2:
        return 0.0, 0.0, 0.0
    points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals = np.sort(np.real(np.linalg.eigvals(cov)))[::-1]
    major = float(math.sqrt(max(eigvals[0], 0.0))) * 4.0 if eigvals.size > 0 else 0.0
    minor = float(math.sqrt(max(eigvals[1], 0.0))) * 4.0 if eigvals.size > 1 else 0.0
    eccentricity = 0.0
    if major > EPS:
        ratio = min((minor * minor) / max(major * major, EPS), 1.0)
        eccentricity = float(math.sqrt(max(0.0, 1.0 - ratio)))
    return major, minor, eccentricity


def _touch_border_stats(binary_mask):
    mask = (np.asarray(binary_mask) > 0)
    h, w = mask.shape
    top = int(mask[0, :].any()) if h > 0 else 0
    bottom = int(mask[-1, :].any()) if h > 0 else 0
    left = int(mask[:, 0].any()) if w > 0 else 0
    right = int(mask[:, -1].any()) if w > 0 else 0
    border_pixels = 0
    if h > 0 and w > 0:
        border_pixels = int(mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum())
        border_pixels -= int(mask[0, 0]) + int(mask[0, -1]) + int(mask[-1, 0]) + int(mask[-1, -1])
    return {
        "touch_border_top": top,
        "touch_border_bottom": bottom,
        "touch_border_left": left,
        "touch_border_right": right,
        "touch_border_count": top + bottom + left + right,
        "touch_border_pixels": border_pixels,
    }


def _entropy(prob):
    p = np.clip(np.asarray(prob, dtype=np.float32), EPS, 1.0 - EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def extract_mask_features(prob, binary_mask, high_conf_threshold=0.9, uncertainty_lower=0.4, uncertainty_upper=0.6):
    prob = np.asarray(prob, dtype=np.float32)
    mask = (np.asarray(binary_mask) > 0).astype(np.uint8)
    h, w = mask.shape
    total_pixels = int(mask.size)
    positive_pixels = int(mask.sum())
    positive_fraction = float(positive_pixels / max(total_pixels, 1))

    labeled, components = connected_components_from_binary(mask)
    num_components = len(components)
    largest = components[0] if components else None
    largest_mask = largest["mask"].astype(np.uint8) if largest else np.zeros_like(mask)
    largest_area = int(largest["area"]) if largest else 0
    largest_fraction = float(largest_area / max(total_pixels, 1))
    largest_ratio = float(largest_area / max(positive_pixels, 1)) if positive_pixels > 0 else 0.0
    total_component_area = int(sum(c["area"] for c in components))
    mean_component_area = float(total_component_area / num_components) if num_components else 0.0
    small_component_area = int(sum(c["area"] for c in components[1:]))
    small_component_ratio = float(small_component_area / max(positive_pixels, 1)) if positive_pixels > 0 else 0.0

    if largest:
        x0, y0, x1, y1 = largest["bbox"]
        bbox_width = int(x1 - x0 + 1)
        bbox_height = int(y1 - y0 + 1)
        bbox_area = int(bbox_width * bbox_height)
        bbox_aspect_ratio = float(max(bbox_width, bbox_height) / max(min(bbox_width, bbox_height), 1))
        extent = float(largest_area / max(bbox_area, 1))
    else:
        x0 = y0 = x1 = y1 = -1
        bbox_width = bbox_height = bbox_area = 0
        bbox_aspect_ratio = 0.0
        extent = 0.0

    perimeter = _perimeter(largest_mask)
    compactness = float((perimeter ** 2) / max(4.0 * math.pi * largest_area, EPS)) if largest_area > 0 else 0.0
    hull_area = _convex_hull_area(largest_mask)
    solidity = float(largest_area / max(hull_area, EPS)) if largest_area > 0 else 0.0
    major_axis_length, minor_axis_length, eccentricity = _principal_axes(largest_mask)

    holes_filled = ndimage.binary_fill_holes(largest_mask > 0)
    hole_pixels = int(holes_filled.sum() - largest_mask.sum())
    hole_ratio = float(hole_pixels / max(largest_area, 1)) if largest_area > 0 else 0.0
    hole_labeled, hole_count = ndimage.label((holes_filled.astype(np.uint8) - largest_mask) > 0)

    border = _touch_border_stats(largest_mask)
    touch_border_ratio = float(border["touch_border_pixels"] / max(largest_area, 1)) if largest_area > 0 else 0.0

    prob_flat = prob.reshape(-1)
    entropy = _entropy(prob)
    high_conf_mask = prob >= float(high_conf_threshold)
    uncertain_mask = (prob >= float(uncertainty_lower)) & (prob <= float(uncertainty_upper))

    if positive_pixels > 0:
        fg_prob = prob[mask > 0]
        fg_entropy = entropy[mask > 0]
        high_conf_fg_pixels = int(((prob >= float(high_conf_threshold)) & (mask > 0)).sum())
    else:
        fg_prob = np.array([], dtype=np.float32)
        fg_entropy = np.array([], dtype=np.float32)
        high_conf_fg_pixels = 0

    return {
        "image_height": h,
        "image_width": w,
        "num_pixels": total_pixels,
        "pred_has_positive": int(positive_pixels > 0),
        "pred_positive_pixels": positive_pixels,
        "pred_positive_fraction": positive_fraction,
        "area_ratio": positive_fraction,
        "num_components": num_components,
        "component_count": num_components,
        "total_component_area": total_component_area,
        "largest_component_area": largest_area,
        "largest_component_fraction": largest_fraction,
        "largest_component_area_ratio": largest_fraction,
        "largest_component_ratio": largest_ratio,
        "mean_component_area": mean_component_area,
        "small_component_ratio": small_component_ratio,
        "bbox_x0": x0,
        "bbox_y0": y0,
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_area,
        "bbox_aspect_ratio": bbox_aspect_ratio,
        "extent": extent,
        "perimeter": perimeter,
        "compactness": compactness,
        "solidity": solidity,
        "major_axis_length": major_axis_length,
        "minor_axis_length": minor_axis_length,
        "eccentricity": eccentricity,
        "hole_count": int(hole_count),
        "hole_ratio": hole_ratio,
        **border,
        "touch_border_ratio": touch_border_ratio,
        "prob_min": float(prob_flat.min()) if prob_flat.size else 0.0,
        "prob_max": float(prob_flat.max()) if prob_flat.size else 0.0,
        "prob_mean": float(prob_flat.mean()) if prob_flat.size else 0.0,
        "prob_std": float(prob_flat.std()) if prob_flat.size else 0.0,
        "prob_p95": _safe_percentile(prob_flat, 95),
        "prob_p99": _safe_percentile(prob_flat, 99),
        "global_mean_prob": float(prob_flat.mean()) if prob_flat.size else 0.0,
        "global_max_prob": float(prob_flat.max()) if prob_flat.size else 0.0,
        "global_std_prob": float(prob_flat.std()) if prob_flat.size else 0.0,
        "fg_prob_mean": float(fg_prob.mean()) if fg_prob.size else 0.0,
        "fg_mean_prob": float(fg_prob.mean()) if fg_prob.size else 0.0,
        "fg_prob_median": float(np.median(fg_prob)) if fg_prob.size else 0.0,
        "fg_median_prob": float(np.median(fg_prob)) if fg_prob.size else 0.0,
        "fg_prob_max": float(fg_prob.max()) if fg_prob.size else 0.0,
        "fg_max_prob": float(fg_prob.max()) if fg_prob.size else 0.0,
        "fg_prob_std": float(fg_prob.std()) if fg_prob.size else 0.0,
        "fg_std_prob": float(fg_prob.std()) if fg_prob.size else 0.0,
        "fg_prob_p95": _safe_percentile(fg_prob, 95),
        "high_conf_pixels_0p9": int(high_conf_mask.sum()),
        "high_conf_fraction_0p9": float(high_conf_mask.mean()) if high_conf_mask.size else 0.0,
        "high_conf_fg_pixels": high_conf_fg_pixels,
        "high_conf_fg_ratio": float(high_conf_fg_pixels / max(positive_pixels, 1)) if positive_pixels > 0 else 0.0,
        "uncertainty_ratio": float(uncertain_mask.mean()) if uncertain_mask.size else 0.0,
        "entropy_mean": float(entropy.mean()) if entropy.size else 0.0,
        "entropy_fg_mean": float(fg_entropy.mean()) if fg_entropy.size else 0.0,
    }


def compute_screen_score(features: Dict, weights=None):
    if weights is None:
        weights = {"prob_max": 0.25, "fg_prob_mean": 0.25, "largest_component_ratio": 0.20, "pred_positive_fraction": 0.15, "high_conf_fg_ratio": 0.15}

    positive_fraction_score = min(max(features.get("pred_positive_fraction", 0.0) * 5.0, 0.0), 1.0)
    raw = (
        weights.get("prob_max", 0.0) * features.get("prob_max", 0.0)
        + weights.get("fg_prob_mean", 0.0) * features.get("fg_prob_mean", 0.0)
        + weights.get("largest_component_ratio", 0.0) * features.get("largest_component_ratio", 0.0)
        + weights.get("pred_positive_fraction", 0.0) * positive_fraction_score
        + weights.get("high_conf_fg_ratio", 0.0) * features.get("high_conf_fg_ratio", 0.0)
    )
    penalty = 0.0
    if features.get("num_components", 0) > 1:
        penalty += min(0.15, 0.03 * (features.get("num_components", 0) - 1))
    penalty += min(0.20, features.get("uncertainty_ratio", 0.0) * 0.3)
    final = float(max(0.0, min(1.0, raw - penalty)))
    return {"screen_score_raw": float(raw), "screen_score_penalty": float(penalty), "screen_score": final}
