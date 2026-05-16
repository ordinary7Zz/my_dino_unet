import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dino_unet import DINOv3_S_UNet


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RANKING_CHOICES = ["screen_score", "score_max_prob", "score_area", "score_fg_mean"]


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


def patient_id_from_relative_path(relative_path):
    parts = Path(relative_path).parts
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    if parts:
        return parts[0]
    return relative_path


def load_patient_filter_from_json(json_path):
    path = Path(json_path)
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if not isinstance(payload, list):
        raise ValueError("--patient_filter_json must point to a JSON list")

    image_relative_paths = set()
    patient_ids = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        if not filename:
            continue
        normalized = str(filename).replace("\\", "/").strip("/")
        path_parts = Path(normalized).parts
        if not path_parts:
            continue
        rel_path = "/".join(path_parts)
        image_relative_paths.add(rel_path)
        patient_id = patient_id_from_relative_path(rel_path)
        if patient_id:
            patient_ids.add(patient_id)

    if not image_relative_paths:
        raise ValueError("No image paths could be extracted from --patient_filter_json")

    return {
        "image_relative_paths": image_relative_paths,
        "patient_ids": patient_ids,
    }


def find_images_recursive(input_dir, allowed_relative_paths=None):
    root = Path(input_dir)
    image_paths = [
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if allowed_relative_paths is not None:
        image_paths = [
            p for p in image_paths if p.relative_to(root).as_posix() in allowed_relative_paths
        ]
    return sorted(image_paths)


class RecursiveInferenceDataset(Dataset):
    def __init__(self, input_dir, img_size, allowed_relative_paths=None):
        self.input_root = Path(input_dir)
        self.image_paths = find_images_recursive(self.input_root, allowed_relative_paths=allowed_relative_paths)
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


def compute_screening_stats(prob_map, binary_mask, threshold, min_region_area):
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
    stats["score_max_prob"] = stats["prob_max"]
    stats["score_area"] = stats["pred_positive_fraction"]
    stats["score_fg_mean"] = stats["fg_prob_mean"]
    stats["screen_score"] = float(stats["fg_prob_mean"] * stats["largest_component_fraction"])
    return stats


def create_overlay(source_path, binary_mask, output_size):
    with Image.open(source_path) as img:
        base_image = img.convert("RGB")

    if base_image.size != (output_size[1], output_size[0]):
        base_image = base_image.resize((output_size[1], output_size[0]), resample=Image.BILINEAR)

    base_array = np.array(base_image, dtype=np.uint8)
    overlay_array = base_array.copy()
    overlay_array[binary_mask] = np.array([255, 0, 0], dtype=np.uint8)
    blended = (0.65 * base_array + 0.35 * overlay_array).astype(np.uint8)
    return blended


def save_csv(path, rows, fieldnames):
    ensure_parent(path)
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path, payload):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def is_trustworthy_mask(row, args):
    passed = []
    failed = []

    checks = [
        (bool(row["pred_has_positive"]), "pred_has_positive"),
        (row["fg_prob_mean"] >= args.trust_fg_prob_mean_min, f"fg_prob_mean>={args.trust_fg_prob_mean_min}"),
        (row["prob_max"] >= args.trust_prob_max_min, f"prob_max>={args.trust_prob_max_min}"),
        (
            row["largest_component_area"] >= args.trust_largest_component_area_min,
            f"largest_component_area>={args.trust_largest_component_area_min}",
        ),
        (
            row["pred_positive_fraction"] >= args.trust_positive_fraction_min,
            f"pred_positive_fraction>={args.trust_positive_fraction_min}",
        ),
        (
            row["largest_component_ratio"] >= args.trust_largest_component_ratio_min,
            f"largest_component_ratio>={args.trust_largest_component_ratio_min}",
        ),
        (row["num_components"] <= args.trust_num_components_max, f"num_components<={args.trust_num_components_max}"),
        (
            row["high_conf_fraction_0p9"] >= args.trust_high_conf_fraction_0p9_min,
            f"high_conf_fraction_0p9>={args.trust_high_conf_fraction_0p9_min}",
        ),
    ]

    for ok, label in checks:
        if ok:
            passed.append(label)
        else:
            failed.append(label)

    return len(failed) == 0, passed, failed


def patient_id_from_relative_path(relative_path):
    parts = Path(relative_path).parts
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    if parts:
        return parts[0]
    return relative_path



def primary_max_per_patient(value):
    if value < 1:
        raise ValueError("--primary_topk must be within 1-3 for patient-level primary lesion export")
    if value > 3:
        raise ValueError("--primary_topk must be within 1-3 for patient-level primary lesion export")
    return int(value)



def build_primary_lesion_record(row):
    record = {
        "source_path": row["source_path"],
        "relative_path": row["relative_path"],
        "selection_label": "likely_primary_lesion",
        "trustworthy_mask": True,
        "trust_reasons_passed": list(row.get("trust_reasons_passed", [])),
        "scores": {
            "screen_score": float(row["screen_score"]),
            "score_max_prob": float(row["score_max_prob"]),
            "score_area": float(row["score_area"]),
            "score_fg_mean": float(row["score_fg_mean"]),
        },
        "mask_stats": {
            "pred_positive_fraction": float(row["pred_positive_fraction"]),
            "pred_positive_pixels": int(row["pred_positive_pixels"]),
            "num_components": int(row["num_components"]),
            "largest_component_area": int(row["largest_component_area"]),
            "largest_component_fraction": float(row["largest_component_fraction"]),
            "largest_component_ratio": float(row["largest_component_ratio"]),
            "fg_prob_mean": float(row["fg_prob_mean"]),
            "fg_prob_max": float(row["fg_prob_max"]),
            "fg_prob_p95": float(row["fg_prob_p95"]),
            "prob_max": float(row["prob_max"]),
            "high_conf_fraction_0p9": float(row["high_conf_fraction_0p9"]),
        },
        "artifacts": {
            "binary_mask_path": row["binary_mask_path"],
            "prob_png_path": row["prob_png_path"],
            "prob_npy_path": row["prob_npy_path"],
            "overlay_path": row["overlay_path"],
        },
        "bbox": {
            "xmin": int(row["largest_component_bbox_xmin"]),
            "ymin": int(row["largest_component_bbox_ymin"]),
            "xmax": int(row["largest_component_bbox_xmax"]),
            "ymax": int(row["largest_component_bbox_ymax"]),
        },
    }
    if "rank_within_patient" in row:
        record["rank_within_patient"] = int(row["rank_within_patient"])
    return record


def build_rejected_record(row):
    return {
        "source_path": row["source_path"],
        "relative_path": row["relative_path"],
        "trustworthy_mask": False,
        "trust_reasons_failed": list(row.get("trust_reasons_failed", [])),
        "scores": {
            "screen_score": float(row["screen_score"]),
            "score_max_prob": float(row["score_max_prob"]),
            "score_area": float(row["score_area"]),
            "score_fg_mean": float(row["score_fg_mean"]),
        },
        "artifacts": {
            "binary_mask_path": row["binary_mask_path"],
            "prob_png_path": row["prob_png_path"],
            "prob_npy_path": row["prob_npy_path"],
            "overlay_path": row["overlay_path"],
        },
    }


def export_primary_lesion_json(rows, args, tables_dir):
    trusted_rows = []
    rejected_rows = []
    rows_by_patient = {}

    for row in rows:
        trustworthy, passed, failed = is_trustworthy_mask(row, args)
        annotated_row = dict(row)
        annotated_row["trustworthy_mask"] = trustworthy
        annotated_row["trust_reasons_passed"] = passed
        annotated_row["trust_reasons_failed"] = failed
        annotated_row["patient_id"] = patient_id_from_relative_path(annotated_row["relative_path"])
        rows_by_patient.setdefault(annotated_row["patient_id"], []).append(annotated_row)
        if trustworthy:
            trusted_rows.append(annotated_row)
        else:
            rejected_rows.append(annotated_row)

    selected_rows = []
    exported_patient_ids = set()
    patient_limit = primary_max_per_patient(args.primary_topk)
    patient_summary_rows = []

    for patient_id, patient_rows in rows_by_patient.items():
        patient_rows_sorted = sorted(patient_rows, key=lambda item: item[args.primary_sort_score], reverse=True)
        patient_trusted_rows = [row for row in patient_rows_sorted if row["trustworthy_mask"]]
        if patient_trusted_rows:
            patient_selected_rows = patient_trusted_rows[:patient_limit]
        else:
            patient_selected_rows = patient_rows_sorted[:1]
        if patient_selected_rows:
            exported_patient_ids.add(patient_id)

        best_row = patient_rows_sorted[0] if patient_rows_sorted else None
        best_trusted_row = patient_trusted_rows[0] if patient_trusted_rows else None

        patient_summary_rows.append(
            {
                "patient_id": patient_id,
                "num_images": int(len(patient_rows)),
                "num_trusted_images": int(len(patient_trusted_rows)),
                "num_exported_images": int(len(patient_selected_rows)),
                "has_trusted_images": bool(len(patient_trusted_rows) > 0),
                "best_image_relative_path": best_row["relative_path"] if best_row else "",
                "best_image_score": float(best_row[args.primary_sort_score]) if best_row else 0.0,
                "best_trusted_relative_path": best_trusted_row["relative_path"] if best_trusted_row else "",
                "best_trusted_score": float(best_trusted_row[args.primary_sort_score]) if best_trusted_row else 0.0,
                "exported_relative_paths": "|".join(row["relative_path"] for row in patient_selected_rows),
            }
        )

        for patient_rank, row in enumerate(patient_selected_rows, start=1):
            selected_row = dict(row)
            selected_row["rank_within_patient"] = patient_rank
            selected_rows.append(selected_row)

    selected_rows = sorted(selected_rows, key=lambda item: item[args.primary_sort_score], reverse=True)
    patient_summary_rows = sorted(
        patient_summary_rows,
        key=lambda item: (item["num_exported_images"], item["best_trusted_score"], item["best_image_score"]),
        reverse=True,
    )
    patients_with_trusted = {row["patient_id"] for row in trusted_rows}

    payload = {
        "version": "1.0",
        "generated_from": {
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "checkpoint": args.checkpoint,
            "ranking_score": args.primary_sort_score,
            "threshold": float(args.threshold),
            "min_region_area": int(args.min_region_area),
        },
        "selection_config": {
            "trust_fg_prob_mean_min": float(args.trust_fg_prob_mean_min),
            "trust_prob_max_min": float(args.trust_prob_max_min),
            "trust_largest_component_area_min": int(args.trust_largest_component_area_min),
            "trust_positive_fraction_min": float(args.trust_positive_fraction_min),
            "trust_positive_fraction_max": float(args.trust_positive_fraction_max),
            "trust_largest_component_ratio_min": float(args.trust_largest_component_ratio_min),
            "trust_num_components_max": int(args.trust_num_components_max),
            "trust_high_conf_fraction_0p9_min": float(args.trust_high_conf_fraction_0p9_min),
            "primary_topk": int(args.primary_topk),
            "primary_max_per_patient": int(patient_limit),
        },
        "summary": {
            "num_images_total": int(len(rows)),
            "num_trusted": int(len(trusted_rows)),
            "num_exported": int(len(selected_rows)),
            "num_patients_total": int(len(rows_by_patient)),
            "num_patients_with_trusted_images": int(len(patients_with_trusted)),
            "num_patients_exported": int(len(exported_patient_ids)),
        },
        "images": [build_primary_lesion_record(row) for row in selected_rows],
    }

    primary_json_path = tables_dir / args.primary_json_name
    save_json(primary_json_path, payload)

    patient_summary_csv = tables_dir / "patient_summary.csv"
    patient_summary_fieldnames = [
        "patient_id",
        "num_images",
        "num_trusted_images",
        "num_exported_images",
        "has_trusted_images",
        "best_image_relative_path",
        "best_image_score",
        "best_trusted_relative_path",
        "best_trusted_score",
        "exported_relative_paths",
    ]
    save_csv(patient_summary_csv, patient_summary_rows, patient_summary_fieldnames)

    rejected_json_path = None
    if args.export_rejected_json:
        rejected_payload = {
            "version": "1.0",
            "generated_from": payload["generated_from"],
            "selection_config": payload["selection_config"],
            "summary": {
                "num_images_total": int(len(rows)),
                "num_rejected": int(len(rejected_rows)),
            },
            "images": [build_rejected_record(row) for row in rejected_rows],
        }
        rejected_json_path = tables_dir / "rejected_mask_images.json"
        save_json(rejected_json_path, rejected_payload)

    return primary_json_path, rejected_json_path, payload["summary"]


def main():
    parser = argparse.ArgumentParser("DINOv3-UNet Recursive Screening Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--img_size", type=int, default=224, help="Model input image size")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default=None, help='Device, e.g. "cuda", "cuda:0", "cpu"')
    parser.add_argument(
        "--dino_pretrained",
        type=str,
        default="false",
        help="Whether to initialize DINO backbone with pretrained weights (true/false)",
    )
    parser.add_argument(
        "--use_dilation",
        type=str,
        default="false",
        help="Whether to enable dilation block in model (true/false)",
    )
    parser.add_argument(
        "--save_orig_size",
        type=str,
        default="true",
        help="Resize exported predictions back to each image original size before saving (true/false)",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask export")
    parser.add_argument("--save_binary_mask", type=str, default="true", help="Save thresholded mask PNG")
    parser.add_argument("--save_prob_png", type=str, default="true", help="Save probability PNG")
    parser.add_argument("--save_prob_npy", type=str, default="true", help="Save raw probability NPY")
    parser.add_argument("--save_overlay", type=str, default="true", help="Save overlay PNG")
    parser.add_argument("--save_input_copy", type=str, default="false", help="Copy original image to outputs")
    parser.add_argument(
        "--min_region_area",
        type=int,
        default=32,
        help="Ignore connected components smaller than this area when computing component stats",
    )
    parser.add_argument(
        "--ranking_score",
        type=str,
        default="screen_score",
        choices=RANKING_CHOICES,
        help="Column used to sort screening_summary.csv",
    )
    parser.add_argument(
        "--topk_summary",
        type=int,
        default=100,
        help="Write top-k ranked rows to topk_summary.csv; set <=0 to skip",
    )
    parser.add_argument(
        "--export_primary_lesion_json",
        type=str,
        default="false",
        help="Export trusted original images as likely primary lesion JSON (true/false)",
    )
    parser.add_argument(
        "--primary_json_name",
        type=str,
        default="likely_primary_lesion_images.json",
        help="Output JSON filename under tables/ for trusted image export",
    )
    parser.add_argument(
        "--primary_topk",
        type=int,
        default=1,
        help="Max trusted images to export per patient (must be within 1-3; if no trusted images exist, export the best-scoring image)",
    )
    parser.add_argument(
        "--primary_sort_score",
        type=str,
        default="screen_score",
        choices=RANKING_CHOICES,
        help="Score column used to sort trusted images before JSON export",
    )
    parser.add_argument("--trust_fg_prob_mean_min", type=float, default=0.60)
    parser.add_argument("--trust_prob_max_min", type=float, default=0.80)
    parser.add_argument("--trust_largest_component_area_min", type=int, default=64)
    parser.add_argument("--trust_positive_fraction_min", type=float, default=0.001)
    parser.add_argument("--trust_positive_fraction_max", type=float, default=0.35)
    parser.add_argument("--trust_largest_component_ratio_min", type=float, default=0.50)
    parser.add_argument("--trust_num_components_max", type=int, default=5)
    parser.add_argument("--trust_high_conf_fraction_0p9_min", type=float, default=0.0)
    parser.add_argument(
        "--patient_filter_json",
        type=str,
        default="",
        help="Optional JSON file; if provided, only patients appearing in its filename fields will be processed",
    )
    parser.add_argument(
        "--export_rejected_json",
        type=str,
        default="false",
        help="Export rejected-image JSON with failure reasons (true/false)",
    )
    args = parser.parse_args()

    args.input_dir = clean_path(args.input_dir)
    args.output_dir = clean_path(args.output_dir)
    args.checkpoint = clean_path(args.checkpoint)
    args.patient_filter_json = clean_path(args.patient_filter_json)
    args.dino_pretrained = str2bool(args.dino_pretrained)
    args.use_dilation = str2bool(args.use_dilation)
    args.save_orig_size = str2bool(args.save_orig_size)
    args.save_binary_mask = str2bool(args.save_binary_mask)
    args.save_prob_png = str2bool(args.save_prob_png)
    args.save_prob_npy = str2bool(args.save_prob_npy)
    args.save_overlay = str2bool(args.save_overlay)
    args.save_input_copy = str2bool(args.save_input_copy)
    args.export_primary_lesion_json = str2bool(args.export_primary_lesion_json)
    args.export_rejected_json = str2bool(args.export_rejected_json)

    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"--input_dir must be an existing directory: {args.input_dir}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    if args.patient_filter_json and not os.path.isfile(args.patient_filter_json):
        raise FileNotFoundError(f"Patient filter JSON not found: {args.patient_filter_json}")
    if args.min_region_area < 1:
        raise ValueError("--min_region_area must be >= 1")

    patient_filter = None
    if args.patient_filter_json:
        patient_filter = load_patient_filter_from_json(args.patient_filter_json)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    if args.patient_filter_json:
        print(f"Patient filter JSON: {args.patient_filter_json}")
        print(f"Allowed patients: {len(patient_filter['patient_ids'])}")
        print(f"Allowed images: {len(patient_filter['image_relative_paths'])}")

    dataset = RecursiveInferenceDataset(
        args.input_dir,
        args.img_size,
        allowed_relative_paths=(patient_filter["image_relative_paths"] if patient_filter else None),
    )
    if len(dataset) == 0:
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

    start_time = time.time()
    total = len(dataset)
    done = 0
    rows = []

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
                prob_uint8 = probability_to_uint8(prob_map)
                binary_uint8 = binary_to_uint8(binary_mask)
                stats = compute_screening_stats(prob_map, binary_mask, args.threshold, args.min_region_area)

                binary_mask_path = ""
                prob_png_path = ""
                prob_npy_path = ""
                overlay_path = ""
                input_copy_path = ""

                if args.save_binary_mask:
                    binary_path = build_output_path(args.output_dir, "binary_masks", rel_path, ".png")
                    ensure_parent(binary_path)
                    imageio.imsave(binary_path.as_posix(), binary_uint8)
                    binary_mask_path = binary_path.relative_to(args.output_dir).as_posix()

                if args.save_prob_png:
                    prob_png = build_output_path(args.output_dir, "prob_png", rel_path, ".png")
                    ensure_parent(prob_png)
                    imageio.imsave(prob_png.as_posix(), prob_uint8)
                    prob_png_path = prob_png.relative_to(args.output_dir).as_posix()

                if args.save_prob_npy:
                    prob_npy = build_output_path(args.output_dir, "prob_npy", rel_path, ".npy")
                    ensure_parent(prob_npy)
                    np.save(prob_npy.as_posix(), prob_map.astype(np.float32))
                    prob_npy_path = prob_npy.relative_to(args.output_dir).as_posix()

                if args.save_overlay:
                    overlay_png = build_output_path(args.output_dir, "overlays", rel_path, ".png")
                    ensure_parent(overlay_png)
                    overlay_image = create_overlay(source_path, binary_mask, (export_h, export_w))
                    imageio.imsave(overlay_png.as_posix(), overlay_image)
                    overlay_path = overlay_png.relative_to(args.output_dir).as_posix()

                if args.save_input_copy:
                    input_copy = build_output_path(args.output_dir, "inputs", rel_path, None)
                    ensure_parent(input_copy)
                    shutil.copy2(source_path, input_copy)
                    input_copy_path = input_copy.relative_to(args.output_dir).as_posix()

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
                    "input_copy_path": input_copy_path,
                }
                row.update(stats)
                rows.append(row)

                done += 1
                if done % 50 == 0 or done == total:
                    print(f"Progress: {done}/{total}")

    per_image_fieldnames = [
        "relative_path",
        "source_path",
        "orig_width",
        "orig_height",
        "export_width",
        "export_height",
        "binary_mask_path",
        "prob_png_path",
        "prob_npy_path",
        "overlay_path",
        "input_copy_path",
        "threshold",
        "prob_min",
        "prob_max",
        "prob_mean",
        "prob_std",
        "prob_p95",
        "prob_p99",
        "pred_positive_pixels",
        "pred_positive_fraction",
        "pred_has_positive",
        "high_conf_pixels_0p9",
        "high_conf_fraction_0p9",
        "fg_prob_mean",
        "fg_prob_max",
        "fg_prob_p95",
        "num_components",
        "total_component_area",
        "largest_component_area",
        "largest_component_fraction",
        "largest_component_ratio",
        "mean_component_area",
        "largest_component_bbox_xmin",
        "largest_component_bbox_ymin",
        "largest_component_bbox_xmax",
        "largest_component_bbox_ymax",
        "score_max_prob",
        "score_area",
        "score_fg_mean",
        "screen_score",
    ]

    tables_dir = Path(args.output_dir) / "tables"
    per_image_csv = tables_dir / "per_image_stats.csv"
    save_csv(per_image_csv, rows, per_image_fieldnames)

    sorted_rows = sorted(rows, key=lambda row: row[args.ranking_score], reverse=True)
    ranked_rows = []
    for rank, row in enumerate(sorted_rows, start=1):
        ranked_row = {"rank": rank}
        ranked_row.update(row)
        ranked_rows.append(ranked_row)

    summary_fieldnames = [
        "rank",
        "relative_path",
        "screen_score",
        "score_max_prob",
        "score_area",
        "score_fg_mean",
        "prob_max",
        "fg_prob_mean",
        "pred_positive_fraction",
        "num_components",
        "largest_component_area",
        "largest_component_fraction",
        "largest_component_ratio",
        "binary_mask_path",
        "prob_png_path",
        "prob_npy_path",
        "overlay_path",
    ]
    summary_rows = [{field: row.get(field, "") for field in summary_fieldnames} for row in ranked_rows]

    screening_summary_csv = tables_dir / "screening_summary.csv"
    save_csv(screening_summary_csv, summary_rows, summary_fieldnames)

    if args.topk_summary > 0:
        topk_csv = tables_dir / "topk_summary.csv"
        save_csv(topk_csv, summary_rows[: args.topk_summary], summary_fieldnames)

    primary_json_path = None
    rejected_json_path = None
    primary_summary = None
    if args.export_primary_lesion_json:
        primary_json_path, rejected_json_path, primary_summary = export_primary_lesion_json(rows, args, tables_dir)

    elapsed = time.time() - start_time
    print(f"Inference finished. Processed {done} images in {elapsed:.2f}s")
    print(f"Per-image stats: {per_image_csv}")
    print(f"Screening summary: {screening_summary_csv}")
    if primary_json_path is not None:
        print(f"Primary lesion JSON: {primary_json_path}")
        print(
            "Trusted/exported images: "
            f"{primary_summary['num_trusted']}/{primary_summary['num_images_total']} trusted, "
            f"{primary_summary['num_exported']} exported"
        )
    if rejected_json_path is not None:
        print(f"Rejected image JSON: {rejected_json_path}")


if __name__ == "__main__":
    main()
