import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import FullDataset
from dino_unet import DINOv3_S_UNet
from utils.metrics import Dice, ECE, HD95


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


def ensure_trailing_sep(path):
    path = clean_path(path)
    if not path.endswith(("/", "\\")):
        path += "/"
    return path


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


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


def resize_gt_mask(gt_mask, output_size):
    tensor = torch.from_numpy(gt_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=output_size, mode="nearest")
    return resized.squeeze(0).squeeze(0).numpy() >= 0.5


def compute_component_stats(binary_mask, min_region_area):
    from scipy import ndimage

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


def compute_screening_stats(prob_map, binary_mask, pred_threshold, min_region_area):
    positive_pixels = int(binary_mask.sum())
    total_pixels = int(binary_mask.size)
    positive_fraction = float(positive_pixels / max(total_pixels, 1))

    stats = {
        "pred_threshold": float(pred_threshold),
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


def validate_dataset_pairing(image_dir, mask_dir, strict_pair_check=True):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    image_files = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    mask_files = sorted([p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])

    if len(image_files) != len(mask_files):
        raise ValueError(f"Image/mask count mismatch: {len(image_files)} images vs {len(mask_files)} masks")

    mismatches = []
    pairs = []
    for img_path, mask_path in zip(image_files, mask_files):
        img_stem = img_path.stem
        mask_stem = mask_path.stem
        if img_stem != mask_stem:
            mismatches.append((img_path.name, mask_path.name))
        pairs.append((img_path.as_posix(), mask_path.as_posix()))

    if mismatches:
        preview = ", ".join([f"{img} != {mask}" for img, mask in mismatches[:10]])
        message = f"Found {len(mismatches)} filename pairing mismatches. Examples: {preview}"
        if strict_pair_check:
            raise ValueError(message)
        print(f"Warning: {message}")

    return pairs


def is_good_case(row, args):
    if args.quality_rule == "dice_only":
        return row["dice"] >= args.quality_dice_min

    good = row["dice"] >= args.quality_dice_min
    if args.quality_hd95_max is not None:
        good = good and row["hd95"] <= args.quality_hd95_max
    if args.quality_ece_max is not None:
        good = good and row["ece"] <= args.quality_ece_max
    return good


def summarize_values(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def derive_thresholds(rows, args):
    good_rows = [row for row in rows if row["is_good_case"]]
    if not good_rows:
        raise ValueError("No good cases found under the current quality rule; cannot derive thresholds.")

    good_df = {
        key: np.asarray([row[key] for row in good_rows], dtype=np.float64)
        for key in [
            "fg_prob_mean",
            "prob_max",
            "largest_component_area",
            "pred_positive_fraction",
            "largest_component_ratio",
            "high_conf_fraction_0p9",
            "num_components",
        ]
    }

    thresholds = {
        "trust_fg_prob_mean_min": round(float(np.quantile(good_df["fg_prob_mean"], args.lower_quantile)), 4),
        "trust_prob_max_min": round(float(np.quantile(good_df["prob_max"], args.lower_quantile)), 4),
        "trust_largest_component_area_min": int(math.ceil(np.quantile(good_df["largest_component_area"], args.lower_quantile))),
        "trust_positive_fraction_min": round(float(np.quantile(good_df["pred_positive_fraction"], args.lower_quantile)), 4),
        "trust_positive_fraction_max": round(float(np.quantile(good_df["pred_positive_fraction"], args.upper_quantile)), 4),
        "trust_largest_component_ratio_min": round(float(np.quantile(good_df["largest_component_ratio"], args.lower_quantile)), 4),
        "trust_num_components_max": int(math.floor(np.quantile(good_df["num_components"], args.upper_quantile))),
        "trust_high_conf_fraction_0p9_min": round(float(np.quantile(good_df["high_conf_fraction_0p9"], args.lower_quantile)), 4),
    }

    feature_specs = [
        ("fg_prob_mean", "min", thresholds["trust_fg_prob_mean_min"], args.lower_quantile),
        ("prob_max", "min", thresholds["trust_prob_max_min"], args.lower_quantile),
        ("largest_component_area", "min", thresholds["trust_largest_component_area_min"], args.lower_quantile),
        ("pred_positive_fraction", "min", thresholds["trust_positive_fraction_min"], args.lower_quantile),
        ("pred_positive_fraction", "max", thresholds["trust_positive_fraction_max"], args.upper_quantile),
        ("largest_component_ratio", "min", thresholds["trust_largest_component_ratio_min"], args.lower_quantile),
        ("num_components", "max", thresholds["trust_num_components_max"], args.upper_quantile),
        ("high_conf_fraction_0p9", "min", thresholds["trust_high_conf_fraction_0p9_min"], args.lower_quantile),
    ]

    feature_rows = []
    for feature_name, direction, threshold_value, quantile_used in feature_specs:
        all_stats = summarize_values(rows, feature_name)
        good_stats = summarize_values(good_rows, feature_name)
        feature_rows.append(
            {
                "feature_name": feature_name,
                "direction": direction,
                "recommended_threshold": threshold_value,
                "quantile_used": float(quantile_used),
                "all_mean": all_stats["mean"],
                "all_median": all_stats["median"],
                "good_mean": good_stats["mean"],
                "good_median": good_stats["median"],
            }
        )

    return thresholds, feature_rows, good_rows


def apply_thresholds(row, thresholds):
    return (
        bool(row["pred_has_positive"])
        and row["fg_prob_mean"] >= thresholds["trust_fg_prob_mean_min"]
        and row["prob_max"] >= thresholds["trust_prob_max_min"]
        and row["largest_component_area"] >= thresholds["trust_largest_component_area_min"]
        and row["pred_positive_fraction"] >= thresholds["trust_positive_fraction_min"]
        and row["pred_positive_fraction"] <= thresholds["trust_positive_fraction_max"]
        and row["largest_component_ratio"] >= thresholds["trust_largest_component_ratio_min"]
        and row["num_components"] <= thresholds["trust_num_components_max"]
        and row["high_conf_fraction_0p9"] >= thresholds["trust_high_conf_fraction_0p9_min"]
    )


def build_validation_summary(rows, thresholds):
    trusted_rows = [row for row in rows if apply_thresholds(row, thresholds)]
    rejected_rows = [row for row in rows if not apply_thresholds(row, thresholds)]
    good_rows = [row for row in rows if row["is_good_case"]]
    trusted_good_rows = [row for row in trusted_rows if row["is_good_case"]]

    trusted_precision = float(len(trusted_good_rows) / len(trusted_rows)) if trusted_rows else 0.0
    good_recall = float(len(trusted_good_rows) / len(good_rows)) if good_rows else 0.0

    return {
        "num_cases": int(len(rows)),
        "num_good_cases": int(len(good_rows)),
        "num_trusted": int(len(trusted_rows)),
        "num_rejected": int(len(rejected_rows)),
        "trusted_precision_for_good_cases": trusted_precision,
        "good_case_recall": good_recall,
        "trusted_dice": summarize_values(trusted_rows, "dice"),
        "rejected_dice": summarize_values(rejected_rows, "dice"),
        "trusted_hd95": summarize_values(trusted_rows, "hd95"),
        "rejected_hd95": summarize_values(rejected_rows, "hd95"),
        "trusted_ece": summarize_values(trusted_rows, "ece"),
        "rejected_ece": summarize_values(rejected_rows, "ece"),
    }


def write_cli_snippet(path, thresholds):
    ensure_parent(path)
    lines = [
        f'--trust_fg_prob_mean_min "{thresholds["trust_fg_prob_mean_min"]}"',
        f'--trust_prob_max_min "{thresholds["trust_prob_max_min"]}"',
        f'--trust_largest_component_area_min "{thresholds["trust_largest_component_area_min"]}"',
        f'--trust_positive_fraction_min "{thresholds["trust_positive_fraction_min"]}"',
        f'--trust_positive_fraction_max "{thresholds["trust_positive_fraction_max"]}"',
        f'--trust_largest_component_ratio_min "{thresholds["trust_largest_component_ratio_min"]}"',
        f'--trust_num_components_max "{thresholds["trust_num_components_max"]}"',
        f'--trust_high_conf_fraction_0p9_min "{thresholds["trust_high_conf_fraction_0p9_min"]}"',
    ]
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(" \\\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser("Recommend trusted-mask thresholds from a labeled lesion dataset")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dino_pretrained", type=str, default="false")
    parser.add_argument("--use_dilation", type=str, default="false")
    parser.add_argument("--save_orig_size", type=str, default="true")
    parser.add_argument("--pred_threshold", type=float, default=0.5)
    parser.add_argument("--min_region_area", type=int, default=32)
    parser.add_argument("--quality_rule", type=str, default="dice_only", choices=["dice_only", "composite"])
    parser.add_argument("--quality_dice_min", type=float, default=0.70)
    parser.add_argument("--quality_hd95_max", type=float, default=None)
    parser.add_argument("--quality_ece_max", type=float, default=None)
    parser.add_argument("--lower_quantile", type=float, default=0.10)
    parser.add_argument("--upper_quantile", type=float, default=0.90)
    parser.add_argument("--strict_pair_check", type=str, default="true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.checkpoint = clean_path(args.checkpoint)
    args.image_dir = ensure_trailing_sep(args.image_dir)
    args.mask_dir = ensure_trailing_sep(args.mask_dir)
    args.output_dir = clean_path(args.output_dir)
    args.dino_pretrained = str2bool(args.dino_pretrained)
    args.use_dilation = str2bool(args.use_dilation)
    args.save_orig_size = str2bool(args.save_orig_size)
    args.strict_pair_check = str2bool(args.strict_pair_check)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    if not os.path.isdir(args.image_dir):
        raise NotADirectoryError(f"Image directory not found: {args.image_dir}")
    if not os.path.isdir(args.mask_dir):
        raise NotADirectoryError(f"Mask directory not found: {args.mask_dir}")
    if args.min_region_area < 1:
        raise ValueError("--min_region_area must be >= 1")
    if not (0.0 <= args.lower_quantile <= 1.0 and 0.0 <= args.upper_quantile <= 1.0):
        raise ValueError("Quantiles must be within [0, 1]")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset_name}")

    pairs = validate_dataset_pairing(args.image_dir, args.mask_dir, strict_pair_check=args.strict_pair_check)
    pair_map = {Path(img).name: {"image_path": img, "mask_path": mask} for img, mask in pairs}

    dataset = FullDataset(args.image_dir, args.mask_dir, args.img_size, mode="test")
    dataloader = DataLoader(
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

    dice_calculator = Dice()
    hd95_calculator = HD95()
    ece_calculator = ECE()

    rows = []
    start_time = time.time()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            filenames = batch["filename"]
            orig_sizes = batch["orig_size"]

            logits = model(images)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = torch.sigmoid(logits).detach().cpu()
            labels_cpu = labels.detach().cpu()

            for i in range(probs.shape[0]):
                filename = filenames[i]
                if filename not in pair_map:
                    raise KeyError(f"Filename {filename} not found in validated image/mask pairs")

                orig_w = int(orig_sizes[0][i].item()) if isinstance(orig_sizes, (list, tuple)) else int(orig_sizes[i][0].item())
                orig_h = int(orig_sizes[1][i].item()) if isinstance(orig_sizes, (list, tuple)) else int(orig_sizes[i][1].item())

                prob_map = probs[i, 0].numpy().astype(np.float32)
                pred_mask = prob_map >= args.pred_threshold
                gt_mask = (labels_cpu[i, 0].numpy() > 0.5)

                if args.save_orig_size and (prob_map.shape[1], prob_map.shape[0]) != (orig_w, orig_h):
                    prob_map = resize_prob_map(prob_map, (orig_h, orig_w)).astype(np.float32)
                    pred_mask = resize_binary_mask(pred_mask, (orig_h, orig_w))
                    gt_mask = resize_gt_mask(gt_mask, (orig_h, orig_w))
                else:
                    pred_mask = pred_mask.astype(bool)
                    gt_mask = gt_mask.astype(bool)

                pred_tensor = torch.from_numpy(pred_mask.astype(np.float32))
                gt_tensor = torch.from_numpy(gt_mask.astype(np.float32))
                prob_tensor = torch.from_numpy(prob_map.astype(np.float32))

                dice = float(dice_calculator(pred_tensor, gt_tensor).item())
                hd95 = float(hd95_calculator(pred_tensor, gt_tensor).item())
                ece = float(ece_calculator(prob_tensor, gt_tensor).item())

                screening_stats = compute_screening_stats(prob_map, pred_mask, args.pred_threshold, args.min_region_area)
                row = {
                    "dataset_name": args.dataset_name,
                    "filename": filename,
                    "image_path": pair_map[filename]["image_path"],
                    "mask_path": pair_map[filename]["mask_path"],
                    "orig_width": orig_w,
                    "orig_height": orig_h,
                    "export_width": int(prob_map.shape[1]),
                    "export_height": int(prob_map.shape[0]),
                    "dice": dice,
                    "hd95": hd95,
                    "ece": ece,
                }
                row.update(screening_stats)
                rows.append(row)

    for row in rows:
        row["is_good_case"] = bool(is_good_case(row, args))

    thresholds, feature_rows, good_rows = derive_thresholds(rows, args)
    validation_summary = build_validation_summary(rows, thresholds)

    recommended_thresholds_payload = {
        "version": "1.0",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": args.checkpoint,
        "dataset_name": args.dataset_name,
        "image_dir": args.image_dir,
        "mask_dir": args.mask_dir,
        "pred_threshold": float(args.pred_threshold),
        "min_region_area": int(args.min_region_area),
        "quality_rule": args.quality_rule,
        "quality_dice_min": float(args.quality_dice_min),
        "quality_hd95_max": args.quality_hd95_max,
        "quality_ece_max": args.quality_ece_max,
        "lower_quantile": float(args.lower_quantile),
        "upper_quantile": float(args.upper_quantile),
        "num_cases": int(len(rows)),
        "num_good_cases": int(len(good_rows)),
        "thresholds": thresholds,
    }

    per_case_csv = Path(args.output_dir) / "per_case_metrics_and_features.csv"
    feature_summary_csv = Path(args.output_dir) / "feature_quality_summary.csv"
    thresholds_json = Path(args.output_dir) / "recommended_thresholds.json"
    validation_json = Path(args.output_dir) / "threshold_validation_summary.json"
    cli_snippet_txt = Path(args.output_dir) / "thresholds_for_infer_recursive_screening.txt"

    per_case_fieldnames = [
        "dataset_name",
        "filename",
        "image_path",
        "mask_path",
        "orig_width",
        "orig_height",
        "export_width",
        "export_height",
        "dice",
        "hd95",
        "ece",
        "is_good_case",
        "pred_threshold",
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

    save_csv(per_case_csv, rows, per_case_fieldnames)
    save_csv(feature_summary_csv, feature_rows, list(feature_rows[0].keys()) if feature_rows else ["feature_name"])
    save_json(thresholds_json, recommended_thresholds_payload)
    save_json(validation_json, validation_summary)
    write_cli_snippet(cli_snippet_txt, thresholds)

    elapsed = time.time() - start_time
    print(f"Finished threshold recommendation in {elapsed:.2f}s")
    print(f"Per-case metrics/features: {per_case_csv}")
    print(f"Recommended thresholds: {thresholds_json}")
    print(f"Validation summary: {validation_json}")
    print(f"CLI snippet: {cli_snippet_txt}")


if __name__ == "__main__":
    main()
