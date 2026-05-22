import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from LNM_screening.features import extract_mask_features
from LNM_screening.io_utils import (
    load_yaml,
    normalize_rel_path,
    rows_to_manifest,
    stringify_rule_list,
    write_csv,
    write_manifest,
)
from LNM_screening.rules import assign_screening_label, load_thresholds


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


CORE_FEATURE_FIELDS = [
    "area_ratio",
    "largest_component_area_ratio",
    "largest_component_ratio",
    "component_count",
    "small_component_ratio",
    "bbox_aspect_ratio",
    "extent",
    "compactness",
    "eccentricity",
    "solidity",
    "touch_border_ratio",
    "hole_ratio",
    "global_mean_prob",
    "global_max_prob",
    "global_std_prob",
    "fg_mean_prob",
    "fg_median_prob",
    "fg_max_prob",
    "fg_std_prob",
    "high_conf_fg_ratio",
    "uncertainty_ratio",
    "entropy_mean",
    "entropy_fg_mean",
]


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
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_binary_mask(path):
    mask = imageio.imread(Path(path).as_posix())
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (np.asarray(mask) > 0).astype(np.uint8)


def load_prob_map(path):
    prob = np.load(Path(path).as_posix()).astype(np.float32)
    return np.clip(prob, 0.0, 1.0)


def build_feature_kwargs(config):
    probability_cfg = config.get("probability", {})
    return {
        "high_conf_threshold": probability_cfg.get("high_conf_threshold", 0.9),
        "uncertainty_lower": probability_cfg.get("uncertainty_lower", 0.4),
        "uncertainty_upper": probability_cfg.get("uncertainty_upper", 0.6),
    }


def build_manifest_dir(args, config):
    manifest_dir = args.manifest_dir or config.get("manifests", {}).get("directory")
    if manifest_dir:
        return Path(clean_path(manifest_dir))
    return Path(args.output_csv).resolve().parent


def build_output_row(image_path, binary_path, prob_path, features, decision):
    row = {
        "image_path": normalize_rel_path(image_path),
        "source_path": normalize_rel_path(image_path),
        "binary_mask_path": normalize_rel_path(binary_path),
        "prob_mask_path": normalize_rel_path(prob_path),
        "present_pred": decision["screening_label"],
        "confidence": decision["confidence"],
        "final_score": decision["screen_score"],
        "score": decision["screen_score"],
        "screen_score_raw": decision["screen_score_raw"],
        "screen_score_penalty": decision["screen_score_penalty"],
        "hard_reject_flag": decision["hard_reject_flag"],
        "trustworthy_mask": decision["trustworthy_mask"],
        "passed_rules": stringify_rule_list(decision["passed_rules"]),
        "failed_rules": stringify_rule_list(decision["failed_rules"]),
        "borderline_rules": stringify_rule_list(decision["borderline_rules"]),
        "reason": decision["reason"],
    }
    row.update(features)
    return row


def write_subset_manifests(rows, manifest_dir, path_key="image_path"):
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for label in ["yes", "suspicious", "no"]:
        subset_rows = [row for row in rows if row.get("present_pred") == label]
        manifest_rows = rows_to_manifest(subset_rows, path_key=path_key)
        write_manifest(manifest_dir / f"{label}_manifest.txt", manifest_rows)


def process_one_case(image_path, image_root, binary_root, prob_root, feature_kwargs, thresholds):
    rel_path = image_path.relative_to(image_root)
    binary_path = binary_root / rel_path.with_suffix(".png")
    prob_path = prob_root / rel_path.with_suffix(".npy")

    if not binary_path.is_file():
        raise FileNotFoundError(f"Binary mask not found: {binary_path}")
    if not prob_path.is_file():
        raise FileNotFoundError(f"Probability map not found: {prob_path}")

    binary_mask = load_binary_mask(binary_path)
    prob_map = load_prob_map(prob_path)
    if binary_mask.shape != prob_map.shape:
        raise ValueError(
            f"Shape mismatch for {image_path}: binary {binary_mask.shape} vs prob {prob_map.shape}"
        )

    features = extract_mask_features(prob_map, binary_mask, **feature_kwargs)
    decision = assign_screening_label(features, thresholds=thresholds)
    return build_output_row(image_path, binary_path, prob_path, features, decision)


def main():
    parser = argparse.ArgumentParser("Lymph Node Region Screening")
    parser.add_argument("--image_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--binary_mask_dir", type=str, required=True, help="Directory of binary masks")
    parser.add_argument("--prob_mask_dir", type=str, required=True, help="Directory of probability .npy files")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--save_subset_manifest", type=str, default="true", help="Whether to save yes/suspicious/no manifests")
    parser.add_argument("--manifest_dir", type=str, default=None, help="Optional manifest output directory")
    parser.add_argument("--copy_mode", type=str, default="none", choices=["none", "manifest"], help="Reserved output mode; manifest writes manifest files only")
    args = parser.parse_args()

    args.image_dir = clean_path(args.image_dir)
    args.binary_mask_dir = clean_path(args.binary_mask_dir)
    args.prob_mask_dir = clean_path(args.prob_mask_dir)
    args.output_csv = clean_path(args.output_csv)
    args.config = clean_path(args.config)
    if args.manifest_dir is not None:
        args.manifest_dir = clean_path(args.manifest_dir)
    args.save_subset_manifest = str2bool(args.save_subset_manifest)

    image_root = Path(args.image_dir)
    binary_root = Path(args.binary_mask_dir)
    prob_root = Path(args.prob_mask_dir)

    if not image_root.is_dir():
        raise NotADirectoryError(f"--image_dir must be an existing directory: {image_root}")
    if not binary_root.is_dir():
        raise NotADirectoryError(f"--binary_mask_dir must be an existing directory: {binary_root}")
    if not prob_root.is_dir():
        raise NotADirectoryError(f"--prob_mask_dir must be an existing directory: {prob_root}")
    if not Path(args.config).is_file():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_yaml(args.config)
    thresholds = load_thresholds(args.config)
    feature_kwargs = build_feature_kwargs(config)

    image_paths = find_images_recursive(image_root)
    if not image_paths:
        print("No image files found under image_dir.")
        return

    rows = []
    for image_path in image_paths:
        rows.append(
            process_one_case(
                image_path=image_path,
                image_root=image_root,
                binary_root=binary_root,
                prob_root=prob_root,
                feature_kwargs=feature_kwargs,
                thresholds=thresholds,
            )
        )

    preferred_fields = [
        "image_path",
        "source_path",
        "binary_mask_path",
        "prob_mask_path",
        "present_pred",
        "confidence",
        "final_score",
        "score",
        "screen_score_raw",
        "screen_score_penalty",
        "hard_reject_flag",
        "trustworthy_mask",
        "passed_rules",
        "failed_rules",
        "borderline_rules",
        "reason",
    ]
    preferred_fields.extend([field for field in CORE_FEATURE_FIELDS if field not in preferred_fields])
    all_row_fields = []
    for row in rows:
        for key in row.keys():
            if key not in all_row_fields:
                all_row_fields.append(key)
    fieldnames = preferred_fields + [field for field in all_row_fields if field not in preferred_fields]

    write_csv(args.output_csv, rows, fieldnames=fieldnames)

    if args.save_subset_manifest or args.copy_mode == "manifest":
        manifest_dir = build_manifest_dir(args, config)
        write_subset_manifests(rows, manifest_dir)

    print(f"Screened {len(rows)} images.")
    print(f"Results CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
