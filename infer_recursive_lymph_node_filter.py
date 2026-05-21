import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dino_unet import DINOv3_S_UNet


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def clean_path(path):
    if isinstance(path, str):
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        path = path.strip()
    return path


def find_images_recursive(input_dir):
    root = Path(input_dir)
    image_paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(image_paths)


class RecursiveInferenceDataset(Dataset):
    def __init__(self, input_dir, img_size):
        self.input_root = Path(input_dir)
        self.image_paths = find_images_recursive(self.input_root)
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            image_tensor = self.transform(img)

        rel_path = image_path.relative_to(self.input_root).as_posix()
        return {
            "image": image_tensor,
            "rel_path": rel_path,
            "source_path": image_path.as_posix(),
            "orig_w": orig_w,
            "orig_h": orig_h,
        }


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if len(state_dict) > 0 and next(iter(state_dict)).startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    return missing_keys, unexpected_keys


def resize_prob_map(prob_map, output_size):
    tensor = torch.from_numpy(prob_map).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=output_size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()


def resize_binary_mask(binary_mask, output_size):
    tensor = torch.from_numpy(binary_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=output_size, mode="nearest")
    return resized.squeeze(0).squeeze(0).numpy() >= 0.5


def probability_to_uint8(prob_map):
    return np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)


def binary_to_uint8(binary_mask):
    return binary_mask.astype(np.uint8) * 255


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def build_output_path(output_dir, subdir, rel_path, suffix=None):
    rel = Path(rel_path)
    if suffix is not None:
        rel = rel.with_suffix(suffix)
    return Path(output_dir) / subdir / rel


def save_json(path, payload):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def compute_component_stats(binary_mask, min_region_area):
    total_pixels = int(binary_mask.size)
    labeled, num_labels = ndimage.label(binary_mask, structure=np.ones((3, 3), dtype=np.uint8))

    components = []
    for label_idx in range(1, num_labels + 1):
        coords = np.argwhere(labeled == label_idx)
        area = int(coords.shape[0])
        if area < min_region_area:
            continue
        ymin, xmin = coords.min(axis=0).tolist()
        ymax, xmax = coords.max(axis=0).tolist()
        components.append(
            {
                "area": area,
                "bbox_xmin": int(xmin),
                "bbox_ymin": int(ymin),
                "bbox_xmax": int(xmax),
                "bbox_ymax": int(ymax),
            }
        )

    if not components:
        return {
            "num_components": 0,
            "total_component_area": 0,
            "largest_component_area": 0,
            "largest_component_fraction": 0.0,
            "largest_component_ratio": 0.0,
            "mean_component_area": 0.0,
            "largest_component_bbox_xmin": -1,
            "largest_component_bbox_ymin": -1,
            "largest_component_bbox_xmax": -1,
            "largest_component_bbox_ymax": -1,
        }

    total_component_area = int(sum(component["area"] for component in components))
    largest_component = max(components, key=lambda item: item["area"])

    return {
        "num_components": len(components),
        "total_component_area": total_component_area,
        "largest_component_area": int(largest_component["area"]),
        "largest_component_fraction": float(largest_component["area"] / max(total_pixels, 1)),
        "largest_component_ratio": float(largest_component["area"] / max(total_component_area, 1)),
        "mean_component_area": float(total_component_area / len(components)),
        "largest_component_bbox_xmin": int(largest_component["bbox_xmin"]),
        "largest_component_bbox_ymin": int(largest_component["bbox_ymin"]),
        "largest_component_bbox_xmax": int(largest_component["bbox_xmax"]),
        "largest_component_bbox_ymax": int(largest_component["bbox_ymax"]),
    }


def compute_probability_stats(prob_map, binary_mask, threshold, min_region_area):
    positive_pixels = int(binary_mask.sum())
    total_pixels = int(binary_mask.size)
    positive_fraction = float(positive_pixels / max(total_pixels, 1))

    stats = {
        "threshold": float(threshold),
        "prob_min": float(prob_map.min()),
        "prob_max": float(prob_map.max()),
        "prob_mean": float(prob_map.mean()),
        "prob_std": float(prob_map.std()),
        "prob_p95": float(np.percentile(prob_map, 95)),
        "prob_p99": float(np.percentile(prob_map, 99)),
        "pred_positive_pixels": positive_pixels,
        "pred_positive_fraction": positive_fraction,
        "pred_has_positive": bool(positive_pixels > 0),
        "high_conf_pixels_0p9": int((prob_map >= 0.9).sum()),
        "high_conf_fraction_0p9": float((prob_map >= 0.9).sum() / max(total_pixels, 1)),
    }

    if positive_pixels > 0:
        fg_probs = prob_map[binary_mask]
        stats.update(
            {
                "fg_prob_mean": float(fg_probs.mean()),
                "fg_prob_max": float(fg_probs.max()),
                "fg_prob_p95": float(np.percentile(fg_probs, 95)),
            }
        )
    else:
        stats.update(
            {
                "fg_prob_mean": 0.0,
                "fg_prob_max": 0.0,
                "fg_prob_p95": 0.0,
            }
        )

    stats.update(compute_component_stats(binary_mask, min_region_area))
    return stats


def load_analysis_images(source_path, output_size):
    with Image.open(source_path) as img:
        rgb = img.convert("RGB")
        gray = img.convert("L")

    target_size = (output_size[1], output_size[0])
    if rgb.size != target_size:
        rgb = rgb.resize(target_size, resample=Image.BILINEAR)
        gray = gray.resize(target_size, resample=Image.BILINEAR)

    rgb_array = np.asarray(rgb, dtype=np.uint8)
    gray_array = np.asarray(gray, dtype=np.uint8)
    return rgb_array, gray_array


def get_largest_component_details(binary_mask, min_region_area):
    height, width = binary_mask.shape
    labeled, num_labels = ndimage.label(binary_mask, structure=np.ones((3, 3), dtype=np.uint8))

    components = []
    for label_idx in range(1, num_labels + 1):
        coords = np.argwhere(labeled == label_idx)
        area = int(coords.shape[0])
        if area < min_region_area:
            continue
        ymin, xmin = coords.min(axis=0).tolist()
        ymax, xmax = coords.max(axis=0).tolist()
        cy, cx = coords.mean(axis=0).tolist()
        mask = labeled == label_idx
        bbox_width = int(xmax - xmin + 1)
        bbox_height = int(ymax - ymin + 1)
        bbox_area = int(bbox_width * bbox_height)
        components.append(
            {
                "area": area,
                "mask": mask,
                "bbox_xmin": int(xmin),
                "bbox_ymin": int(ymin),
                "bbox_xmax": int(xmax),
                "bbox_ymax": int(ymax),
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "bbox_area": bbox_area,
                "bbox_area_fraction": float(bbox_area / max(height * width, 1)),
                "component_fill_ratio": float(area / max(bbox_area, 1)),
                "bbox_aspect_ratio": float(max(bbox_width, bbox_height) / max(min(bbox_width, bbox_height), 1)),
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "centroid_x_norm": float(cx / max(width - 1, 1)),
                "centroid_y_norm": float(cy / max(height - 1, 1)),
                "touches_border": bool(xmin == 0 or ymin == 0 or xmax == width - 1 or ymax == height - 1),
            }
        )

    if not components:
        return {
            "mask": np.zeros_like(binary_mask, dtype=bool),
            "bbox_xmin": -1,
            "bbox_ymin": -1,
            "bbox_xmax": -1,
            "bbox_ymax": -1,
            "bbox_width": 0,
            "bbox_height": 0,
            "bbox_area": 0,
            "bbox_area_fraction": 0.0,
            "component_fill_ratio": 0.0,
            "bbox_aspect_ratio": 0.0,
            "centroid_x": -1.0,
            "centroid_y": -1.0,
            "centroid_x_norm": -1.0,
            "centroid_y_norm": -1.0,
            "distance_to_image_center": 1.0,
            "touches_border": False,
        }

    largest = max(components, key=lambda item: item["area"])
    dx = largest["centroid_x_norm"] - 0.5
    dy = largest["centroid_y_norm"] - 0.5
    largest["distance_to_image_center"] = float(np.sqrt(dx * dx + dy * dy) / np.sqrt(0.5 * 0.5 + 0.5 * 0.5))
    return largest


def compute_valid_ultrasound_mask(gray_array, black_threshold):
    valid_mask = gray_array > black_threshold
    if valid_mask.any():
        valid_mask = ndimage.binary_closing(valid_mask, structure=np.ones((5, 5), dtype=bool), iterations=1)
        valid_mask = ndimage.binary_fill_holes(valid_mask)
    return valid_mask.astype(bool)


def compute_ring_mask(component_mask, ring_width, valid_mask):
    if not component_mask.any():
        return np.zeros_like(component_mask, dtype=bool)
    dilated = ndimage.binary_dilation(component_mask, structure=np.ones((3, 3), dtype=bool), iterations=max(ring_width, 1))
    ring_mask = np.logical_and(dilated, np.logical_not(component_mask))
    ring_mask = np.logical_and(ring_mask, valid_mask)
    return ring_mask


def compute_edge_overlap_score(gray_array, component_mask, valid_mask):
    if not component_mask.any():
        return 0.0

    eroded = ndimage.binary_erosion(component_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    dilated = ndimage.binary_dilation(component_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    boundary_mask = np.logical_and(np.logical_xor(dilated, eroded), valid_mask)
    if not boundary_mask.any():
        return 0.0

    gray_float = gray_array.astype(np.float32)
    grad_y, grad_x = np.gradient(gray_float)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    valid_grad = grad_mag[valid_mask] if valid_mask.any() else grad_mag.reshape(-1)
    if valid_grad.size == 0:
        return 0.0

    grad_norm = float(np.percentile(valid_grad, 95))
    if grad_norm <= 1e-6:
        return 0.0

    boundary_mean = float(grad_mag[boundary_mask].mean())
    return float(np.clip(boundary_mean / grad_norm, 0.0, 1.0))


def compute_image_agreement_features(gray_array, prob_map, component_mask, args):
    valid_mask = compute_valid_ultrasound_mask(gray_array, args.valid_ultrasound_black_threshold)
    ring_mask = compute_ring_mask(component_mask, args.agreement_ring_width, valid_mask)

    if component_mask.any():
        inner_values = gray_array[component_mask].astype(np.float32)
        component_inside_valid_fraction = float(np.logical_and(component_mask, valid_mask).sum() / max(component_mask.sum(), 1))
        component_prob_values = prob_map[component_mask]
        component_prob_mean = float(component_prob_values.mean())
        component_prob_p95 = float(np.percentile(component_prob_values, 95))
        component_high_conf_pixels = int(np.logical_and(component_mask, prob_map >= 0.9).sum())
        high_conf_density = float(component_high_conf_pixels / max(component_mask.sum(), 1))
    else:
        inner_values = np.asarray([], dtype=np.float32)
        component_inside_valid_fraction = 0.0
        component_prob_mean = 0.0
        component_prob_p95 = 0.0
        component_high_conf_pixels = 0
        high_conf_density = 0.0

    if ring_mask.any():
        outer_values = gray_array[ring_mask].astype(np.float32)
    else:
        outer_values = np.asarray([], dtype=np.float32)

    inner_gray_mean = float(inner_values.mean()) if inner_values.size > 0 else 0.0
    inner_gray_std = float(inner_values.std()) if inner_values.size > 0 else 0.0
    outer_gray_mean = float(outer_values.mean()) if outer_values.size > 0 else inner_gray_mean
    outer_gray_std = float(outer_values.std()) if outer_values.size > 0 else 0.0
    contrast_abs_diff = float(abs(inner_gray_mean - outer_gray_mean))
    contrast_signed_diff = float(inner_gray_mean - outer_gray_mean)
    texture_std_ratio = float(inner_gray_std / max(outer_gray_std, 1e-6)) if outer_gray_std > 0 else 0.0
    edge_overlap_score = compute_edge_overlap_score(gray_array, component_mask, valid_mask)
    valid_ultrasound_fraction = float(valid_mask.mean()) if valid_mask.size > 0 else 0.0
    confidence_concentration = float(np.percentile(prob_map, 99) - prob_map.mean())

    return {
        "inner_gray_mean": inner_gray_mean,
        "inner_gray_std": inner_gray_std,
        "outer_gray_mean": outer_gray_mean,
        "outer_gray_std": outer_gray_std,
        "contrast_abs_diff": contrast_abs_diff,
        "contrast_signed_diff": contrast_signed_diff,
        "texture_std_ratio": texture_std_ratio,
        "edge_overlap_score": edge_overlap_score,
        "component_inside_valid_ultrasound_fraction": component_inside_valid_fraction,
        "valid_ultrasound_fraction": valid_ultrasound_fraction,
        "component_prob_mean": component_prob_mean,
        "component_prob_p95": component_prob_p95,
        "component_high_conf_pixels_0p9": component_high_conf_pixels,
        "high_conf_density": high_conf_density,
        "confidence_concentration": confidence_concentration,
    }


def create_overlay(rgb_array, binary_mask):
    overlay_array = rgb_array.copy()
    overlay_array[binary_mask] = np.array([255, 0, 0], dtype=np.uint8)
    return (0.65 * rgb_array + 0.35 * overlay_array).astype(np.uint8)


def clip01(value):
    return float(max(0.0, min(1.0, value)))


def ramp_score(value, low, high):
    if high <= low:
        return 1.0 if value >= high else 0.0
    return clip01((value - low) / (high - low))


def triangular_score(value, left, peak_left, peak_right, right):
    if value <= left or value >= right:
        return 0.0
    if peak_left <= value <= peak_right:
        return 1.0
    if value < peak_left:
        return clip01((value - left) / max(peak_left - left, 1e-6))
    return clip01((right - value) / max(right - peak_right, 1e-6))


def inverse_ramp_score(value, low, high):
    if value <= low:
        return 1.0
    if value >= high:
        return 0.0
    return clip01((high - value) / max(high - low, 1e-6))


def compute_subscores(row, args):
    area_soft_upper = max(args.max_component_fraction * 0.65, args.preferred_component_fraction_peak_high)
    confidence_score = np.mean(
        [
            ramp_score(row["component_prob_mean"], 0.45, 0.85),
            ramp_score(row["component_prob_p95"], 0.60, 0.95),
            ramp_score(row["confidence_concentration"], 0.05, 0.35),
            ramp_score(row["high_conf_density"], 0.01, 0.20),
        ]
    )

    shape_score = np.mean(
        [
            triangular_score(
                row["largest_component_fraction"],
                args.preferred_component_fraction_min * 0.25,
                args.preferred_component_fraction_min,
                area_soft_upper,
                args.max_component_fraction,
            ),
            ramp_score(row["largest_component_ratio"], args.min_largest_component_ratio, 0.95),
            ramp_score(row["component_fill_ratio"], args.min_component_fill_ratio, 0.75),
            triangular_score(row["bbox_aspect_ratio"], 1.0, 1.2, 3.8, 6.0),
        ]
    )

    location_score = np.mean(
        [
            inverse_ramp_score(row["distance_to_image_center"], 0.10, args.max_center_distance),
            0.0 if row["touches_border"] else 1.0,
        ]
    )

    agreement_score = np.mean(
        [
            ramp_score(row["contrast_abs_diff"], 3.0, 18.0),
            ramp_score(row["texture_std_ratio"], 0.60, 1.20),
            ramp_score(row["edge_overlap_score"], 0.10, 0.55),
            ramp_score(row["component_inside_valid_ultrasound_fraction"], 0.60, 0.95),
        ]
    )

    valid_region_score = ramp_score(
        row["component_inside_valid_ultrasound_fraction"],
        args.min_valid_ultrasound_fraction,
        1.0,
    )

    return {
        "confidence": float(confidence_score),
        "shape": float(shape_score),
        "location": float(location_score),
        "agreement": float(agreement_score),
        "valid_region": float(valid_region_score),
    }


def compute_final_score(subscores, args):
    weights = {
        "confidence": args.score_weight_confidence,
        "shape": args.score_weight_shape,
        "location": args.score_weight_location,
        "agreement": args.score_weight_agreement,
        "valid_region": args.score_weight_valid_region,
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        total_weight = 1.0
    score = sum(subscores[key] * weights[key] for key in weights) / total_weight
    return float(score)


def decide_selection(row, args):
    hard_failures = []
    hard_passes = []

    checks = [
        (row["largest_component_area"] >= args.min_component_area, f"largest_component_area>={args.min_component_area}"),
        (row["largest_component_fraction"] <= args.max_component_fraction, f"largest_component_fraction<={args.max_component_fraction}"),
        (row["num_components"] <= args.max_components_reject, f"num_components<={args.max_components_reject}"),
        (
            row["component_inside_valid_ultrasound_fraction"] >= args.min_valid_ultrasound_fraction,
            f"component_inside_valid_ultrasound_fraction>={args.min_valid_ultrasound_fraction}",
        ),
        (
            (not row["touches_border"]) or row["largest_component_fraction"] <= args.border_touch_large_component_fraction,
            f"border_touch_allowed_if_fraction<={args.border_touch_large_component_fraction}",
        ),
    ]

    for ok, label in checks:
        if ok:
            hard_passes.append(label)
        else:
            hard_failures.append(label)

    subscores = compute_subscores(row, args)
    final_score = compute_final_score(subscores, args)

    soft_passes = []
    soft_failures = []
    if final_score >= args.selection_score_threshold:
        soft_passes.append(f"selection_score>={args.selection_score_threshold}")
    else:
        soft_failures.append(f"selection_score>={args.selection_score_threshold}")

    if subscores["agreement"] >= args.min_agreement_score:
        soft_passes.append(f"agreement_score>={args.min_agreement_score}")
    else:
        soft_failures.append(f"agreement_score>={args.min_agreement_score}")

    selected = len(hard_failures) == 0 and len(soft_failures) == 0
    return {
        "selected": bool(selected),
        "selection_score": final_score,
        "subscores": subscores,
        "hard_rejection": bool(len(hard_failures) > 0),
        "passed_rules": hard_passes + soft_passes,
        "failed_rules": hard_failures + soft_failures,
    }


def build_image_record(row, decision, include_artifacts=True):
    record = {
        "source_path": row["source_path"],
        "relative_path": row["relative_path"],
        "selected": bool(decision["selected"]),
        "selection_label": "likely_lymph_node_region" if decision["selected"] else "rejected",
        "selection_score": float(decision["selection_score"]),
        "subscores": {key: float(value) for key, value in decision["subscores"].items()},
        "decision": {
            "hard_rejection": bool(decision["hard_rejection"]),
            "passed_rules": list(decision["passed_rules"]),
            "failed_rules": list(decision["failed_rules"]),
        },
        "mask_stats": {
            "prob_min": float(row["prob_min"]),
            "prob_max": float(row["prob_max"]),
            "prob_mean": float(row["prob_mean"]),
            "prob_std": float(row["prob_std"]),
            "prob_p95": float(row["prob_p95"]),
            "prob_p99": float(row["prob_p99"]),
            "fg_prob_mean": float(row["fg_prob_mean"]),
            "fg_prob_max": float(row["fg_prob_max"]),
            "fg_prob_p95": float(row["fg_prob_p95"]),
            "component_prob_mean": float(row["component_prob_mean"]),
            "component_prob_p95": float(row["component_prob_p95"]),
            "high_conf_fraction_0p9": float(row["high_conf_fraction_0p9"]),
            "high_conf_density": float(row["high_conf_density"]),
            "confidence_concentration": float(row["confidence_concentration"]),
            "pred_positive_pixels": int(row["pred_positive_pixels"]),
            "pred_positive_fraction": float(row["pred_positive_fraction"]),
            "num_components": int(row["num_components"]),
            "largest_component_area": int(row["largest_component_area"]),
            "largest_component_fraction": float(row["largest_component_fraction"]),
            "largest_component_ratio": float(row["largest_component_ratio"]),
            "component_fill_ratio": float(row["component_fill_ratio"]),
            "bbox_area_fraction": float(row["bbox_area_fraction"]),
            "bbox_aspect_ratio": float(row["bbox_aspect_ratio"]),
            "touches_border": bool(row["touches_border"]),
            "distance_to_image_center": float(row["distance_to_image_center"]),
        },
        "image_agreement": {
            "inner_gray_mean": float(row["inner_gray_mean"]),
            "outer_gray_mean": float(row["outer_gray_mean"]),
            "inner_gray_std": float(row["inner_gray_std"]),
            "outer_gray_std": float(row["outer_gray_std"]),
            "contrast_abs_diff": float(row["contrast_abs_diff"]),
            "contrast_signed_diff": float(row["contrast_signed_diff"]),
            "texture_std_ratio": float(row["texture_std_ratio"]),
            "edge_overlap_score": float(row["edge_overlap_score"]),
            "component_inside_valid_ultrasound_fraction": float(row["component_inside_valid_ultrasound_fraction"]),
            "valid_ultrasound_fraction": float(row["valid_ultrasound_fraction"]),
        },
        "bbox": {
            "xmin": int(row["largest_component_bbox_xmin"]),
            "ymin": int(row["largest_component_bbox_ymin"]),
            "xmax": int(row["largest_component_bbox_xmax"]),
            "ymax": int(row["largest_component_bbox_ymax"]),
            "width": int(row["bbox_width"]),
            "height": int(row["bbox_height"]),
        },
        "centroid_norm": {
            "x": float(row["centroid_x_norm"]),
            "y": float(row["centroid_y_norm"]),
        },
    }

    if include_artifacts:
        record["artifacts"] = {
            "binary_mask_path": row["binary_mask_path"],
            "prob_png_path": row["prob_png_path"],
            "prob_npy_path": row["prob_npy_path"],
            "overlay_path": row["overlay_path"],
        }

    return record


def parse_args():
    parser = argparse.ArgumentParser("DINOv3-UNet Recursive Lymph Node Region Filter")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON file")
    parser.add_argument("--img_size", type=int, default=224, help="Model input image size")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default=None, help='Device, e.g. "cuda", "cuda:0", "cpu"')
    parser.add_argument("--dino_pretrained", type=str, default="false")
    parser.add_argument("--use_dilation", type=str, default="false")
    parser.add_argument("--save_orig_size", type=str, default="true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_region_area", type=int, default=32)
    parser.add_argument("--min_component_area", type=int, default=64)
    parser.add_argument("--preferred_component_fraction_min", type=float, default=0.002)
    parser.add_argument("--preferred_component_fraction_peak_high", type=float, default=0.12)
    parser.add_argument("--max_component_fraction", type=float, default=0.25)
    parser.add_argument("--max_components_reject", type=int, default=5)
    parser.add_argument("--min_largest_component_ratio", type=float, default=0.50)
    parser.add_argument("--min_component_fill_ratio", type=float, default=0.20)
    parser.add_argument("--max_center_distance", type=float, default=0.65)
    parser.add_argument("--agreement_ring_width", type=int, default=12)
    parser.add_argument("--valid_ultrasound_black_threshold", type=int, default=8)
    parser.add_argument("--min_valid_ultrasound_fraction", type=float, default=0.80)
    parser.add_argument("--border_touch_large_component_fraction", type=float, default=0.03)
    parser.add_argument("--selection_score_threshold", type=float, default=0.60)
    parser.add_argument("--min_agreement_score", type=float, default=0.30)
    parser.add_argument("--score_weight_confidence", type=float, default=0.25)
    parser.add_argument("--score_weight_shape", type=float, default=0.20)
    parser.add_argument("--score_weight_location", type=float, default=0.15)
    parser.add_argument("--score_weight_agreement", type=float, default=0.30)
    parser.add_argument("--score_weight_valid_region", type=float, default=0.10)
    parser.add_argument("--debug_output_dir", type=str, default="")
    parser.add_argument("--save_binary_mask", type=str, default="false")
    parser.add_argument("--save_prob_png", type=str, default="false")
    parser.add_argument("--save_prob_npy", type=str, default="false")
    parser.add_argument("--save_overlay", type=str, default="false")
    parser.add_argument("--include_rejected_details", type=str, default="false")
    return parser.parse_args()


def main():
    args = parse_args()
    args.input_dir = clean_path(args.input_dir)
    args.output_json = clean_path(args.output_json)
    args.checkpoint = clean_path(args.checkpoint)
    args.debug_output_dir = clean_path(args.debug_output_dir)
    args.dino_pretrained = str2bool(args.dino_pretrained)
    args.use_dilation = str2bool(args.use_dilation)
    args.save_orig_size = str2bool(args.save_orig_size)
    args.save_binary_mask = str2bool(args.save_binary_mask)
    args.save_prob_png = str2bool(args.save_prob_png)
    args.save_prob_npy = str2bool(args.save_prob_npy)
    args.save_overlay = str2bool(args.save_overlay)
    args.include_rejected_details = str2bool(args.include_rejected_details)

    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"--input_dir must be an existing directory: {args.input_dir}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    if args.min_region_area < 1:
        raise ValueError("--min_region_area must be >= 1")
    if args.min_component_area < 1:
        raise ValueError("--min_component_area must be >= 1")

    if any([args.save_binary_mask, args.save_prob_png, args.save_prob_npy, args.save_overlay]):
        if not args.debug_output_dir:
            output_json_path = Path(args.output_json)
            args.debug_output_dir = (output_json_path.parent / f"{output_json_path.stem}_debug").as_posix()
        os.makedirs(args.debug_output_dir, exist_ok=True)

    ensure_parent(Path(args.output_json))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input dir: {args.input_dir}")
    print(f"Output JSON: {args.output_json}")
    if args.debug_output_dir:
        print(f"Debug output dir: {args.debug_output_dir}")

    dataset = RecursiveInferenceDataset(args.input_dir, args.img_size)
    if len(dataset) == 0:
        payload = {
            "version": "1.0",
            "task": "lymph_node_dataset_filter",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generated_from": {
                "input_dir": args.input_dir,
                "checkpoint": args.checkpoint,
                "img_size": int(args.img_size),
                "threshold": float(args.threshold),
            },
            "selection_config": {},
            "summary": {
                "num_images_total": 0,
                "num_selected": 0,
                "num_rejected": 0,
            },
            "images": [],
        }
        save_json(Path(args.output_json), payload)
        print("No image files found under input_dir.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = DINOv3_S_UNet(pretrained=args.dino_pretrained, use_dilation=args.use_dilation).to(device)
    model.eval()

    missing_keys, unexpected_keys = load_checkpoint(model, args.checkpoint, device)
    if missing_keys:
        print(f"Warning: Missing keys when loading checkpoint: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys when loading checkpoint: {len(unexpected_keys)}")

    rows = []
    selected_records = []
    rejected_records = []
    start_time = time.time()
    total = len(dataset)
    done = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            rel_paths = batch["rel_path"]
            source_paths = batch["source_path"]
            orig_ws = batch["orig_w"]
            orig_hs = batch["orig_h"]

            pred = model(images)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]

            probs = torch.sigmoid(pred).detach().cpu()

            for i in range(probs.shape[0]):
                rel_path = rel_paths[i]
                source_path = source_paths[i]
                orig_w = int(orig_ws[i].item())
                orig_h = int(orig_hs[i].item())

                prob_map = probs[i, 0].numpy().astype(np.float32)
                binary_mask = prob_map >= args.threshold

                if args.save_orig_size and (prob_map.shape[1], prob_map.shape[0]) != (orig_w, orig_h):
                    prob_map = resize_prob_map(prob_map, (orig_h, orig_w)).astype(np.float32)
                    binary_mask = resize_binary_mask(binary_mask, (orig_h, orig_w))
                else:
                    binary_mask = binary_mask.astype(bool)

                export_h, export_w = prob_map.shape
                rgb_array, gray_array = load_analysis_images(source_path, (export_h, export_w))
                component_stats = compute_probability_stats(prob_map, binary_mask, args.threshold, args.min_region_area)
                largest_component = get_largest_component_details(binary_mask, args.min_region_area)
                agreement_stats = compute_image_agreement_features(gray_array, prob_map, largest_component["mask"], args)

                binary_mask_path = ""
                prob_png_path = ""
                prob_npy_path = ""
                overlay_path = ""

                if args.debug_output_dir:
                    if args.save_binary_mask:
                        binary_path = build_output_path(args.debug_output_dir, "binary_masks", rel_path, ".png")
                        ensure_parent(binary_path)
                        Image.fromarray(binary_to_uint8(binary_mask)).save(binary_path.as_posix())
                        binary_mask_path = binary_path.relative_to(args.debug_output_dir).as_posix()

                    if args.save_prob_png:
                        prob_png = build_output_path(args.debug_output_dir, "prob_png", rel_path, ".png")
                        ensure_parent(prob_png)
                        Image.fromarray(probability_to_uint8(prob_map)).save(prob_png.as_posix())
                        prob_png_path = prob_png.relative_to(args.debug_output_dir).as_posix()

                    if args.save_prob_npy:
                        prob_npy = build_output_path(args.debug_output_dir, "prob_npy", rel_path, ".npy")
                        ensure_parent(prob_npy)
                        np.save(prob_npy.as_posix(), prob_map.astype(np.float32))
                        prob_npy_path = prob_npy.relative_to(args.debug_output_dir).as_posix()

                    if args.save_overlay:
                        overlay_png = build_output_path(args.debug_output_dir, "overlays", rel_path, ".png")
                        ensure_parent(overlay_png)
                        Image.fromarray(create_overlay(rgb_array, largest_component["mask"])).save(overlay_png.as_posix())
                        overlay_path = overlay_png.relative_to(args.debug_output_dir).as_posix()

                row = {
                    "relative_path": rel_path,
                    "source_path": source_path,
                    "orig_width": orig_w,
                    "orig_height": orig_h,
                    "export_width": int(export_w),
                    "export_height": int(export_h),
                    "binary_mask_path": binary_mask_path,
                    "prob_png_path": prob_png_path,
                    "prob_npy_path": prob_npy_path,
                    "overlay_path": overlay_path,
                }
                row.update(component_stats)
                row.update({k: v for k, v in largest_component.items() if k != "mask"})
                row.update(agreement_stats)

                decision = decide_selection(row, args)
                row["selection_score"] = decision["selection_score"]
                row["selected"] = decision["selected"]
                rows.append(row)

                record = build_image_record(row, decision, include_artifacts=bool(args.debug_output_dir))
                if decision["selected"]:
                    selected_records.append(record)
                elif args.include_rejected_details:
                    rejected_records.append(record)

                done += 1
                if done % 50 == 0 or done == total:
                    print(f"Progress: {done}/{total}")

    selected_records = sorted(selected_records, key=lambda item: item["selection_score"], reverse=True)
    if args.include_rejected_details:
        rejected_records = sorted(rejected_records, key=lambda item: item["selection_score"], reverse=True)

    payload = {
        "version": "1.0",
        "task": "lymph_node_dataset_filter",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_from": {
            "input_dir": args.input_dir,
            "checkpoint": args.checkpoint,
            "img_size": int(args.img_size),
            "threshold": float(args.threshold),
            "debug_output_dir": args.debug_output_dir,
        },
        "selection_config": {
            "min_region_area": int(args.min_region_area),
            "min_component_area": int(args.min_component_area),
            "preferred_component_fraction_min": float(args.preferred_component_fraction_min),
            "preferred_component_fraction_peak_high": float(args.preferred_component_fraction_peak_high),
            "max_component_fraction": float(args.max_component_fraction),
            "max_components_reject": int(args.max_components_reject),
            "min_largest_component_ratio": float(args.min_largest_component_ratio),
            "min_component_fill_ratio": float(args.min_component_fill_ratio),
            "max_center_distance": float(args.max_center_distance),
            "agreement_ring_width": int(args.agreement_ring_width),
            "valid_ultrasound_black_threshold": int(args.valid_ultrasound_black_threshold),
            "min_valid_ultrasound_fraction": float(args.min_valid_ultrasound_fraction),
            "border_touch_large_component_fraction": float(args.border_touch_large_component_fraction),
            "selection_score_threshold": float(args.selection_score_threshold),
            "min_agreement_score": float(args.min_agreement_score),
            "weights": {
                "confidence": float(args.score_weight_confidence),
                "shape": float(args.score_weight_shape),
                "location": float(args.score_weight_location),
                "agreement": float(args.score_weight_agreement),
                "valid_region": float(args.score_weight_valid_region),
            },
        },
        "summary": {
            "num_images_total": int(len(rows)),
            "num_selected": int(len(selected_records)),
            "num_rejected": int(len(rows) - len(selected_records)),
        },
        "images": selected_records,
    }

    if args.include_rejected_details:
        payload["rejected_images"] = rejected_records

    save_json(Path(args.output_json), payload)

    elapsed = time.time() - start_time
    print(f"Inference finished. Processed {done} images in {elapsed:.2f}s")
    print(f"Selected images: {len(selected_records)}/{len(rows)}")
    print(f"Output JSON: {args.output_json}")


if __name__ == "__main__":
    main()
