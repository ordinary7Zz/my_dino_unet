import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_rel_path(path):
    return Path(path).as_posix()


def save_binary_mask(path, mask):
    path = Path(path)
    ensure_dir(path.parent)
    arr = (np.asarray(mask).astype(np.uint8) > 0).astype(np.uint8) * 255
    imageio.imwrite(path.as_posix(), arr)


def save_prob_png(path, prob):
    path = Path(path)
    ensure_dir(path.parent)
    arr = np.clip(np.asarray(prob, dtype=np.float32), 0.0, 1.0)
    imageio.imwrite(path.as_posix(), (arr * 255.0).round().astype(np.uint8))


def save_prob_npy(path, prob):
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path.as_posix(), np.asarray(prob, dtype=np.float32))


def save_overlay(path, image, binary_mask, prob=None, alpha=0.35):
    path = Path(path)
    ensure_dir(path.parent)

    if isinstance(image, Image.Image):
        base = image.convert("RGB")
    else:
        base = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")

    base_arr = np.asarray(base, dtype=np.float32)
    mask = (np.asarray(binary_mask) > 0).astype(np.uint8)
    overlay = base_arr.copy()
    overlay[..., 0] = np.where(mask > 0, 255.0, overlay[..., 0])
    overlay[..., 1] = np.where(mask > 0, overlay[..., 1] * (1.0 - alpha), overlay[..., 1])
    overlay[..., 2] = np.where(mask > 0, overlay[..., 2] * (1.0 - alpha), overlay[..., 2])

    if prob is not None:
        prob_arr = np.clip(np.asarray(prob, dtype=np.float32), 0.0, 1.0)
        heat = np.zeros_like(overlay)
        heat[..., 1] = prob_arr * 255.0
        overlay = np.where(mask[..., None] > 0, overlay, 0.7 * overlay + 0.3 * heat)

    imageio.imwrite(path.as_posix(), np.clip(overlay, 0, 255).astype(np.uint8))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_manifest(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row}\n")


def rows_to_manifest(rows, path_key="source_path"):
    return [row[path_key] for row in rows if row.get(path_key)]


def stringify_rule_list(values: Sequence[str]):
    return "|".join(values) if values else ""
