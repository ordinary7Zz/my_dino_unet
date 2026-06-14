import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            stem = path.stem
            if stem in files:
                raise ValueError(f"发现重复文件名（去掉后缀后同名）：{stem}")
            files[stem] = path
    return files


def count_foreground_pixels(mask_path: Path) -> int:
    with Image.open(mask_path) as img:
        mask = np.asarray(img)

    if mask.ndim == 2:
        return int(np.count_nonzero(mask))

    if mask.ndim == 3:
        return int(np.count_nonzero(np.any(mask != 0, axis=-1)))

    raise ValueError(f"不支持的 mask 维度: {mask_path} -> {mask.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计 mask 中前景像素小于阈值的文件，并保存到 CSV。"
    )
    parser.add_argument("image_dir", type=Path, help="图像目录")
    parser.add_argument("mask_dir", type=Path, help="mask 目录")
    parser.add_argument("output_csv", type=Path, help="输出 CSV 文件路径")
    parser.add_argument("--threshold", type=int, default=10, help="前景像素阈值，默认 10")
    args = parser.parse_args()

    if args.output_csv.exists() and args.output_csv.is_dir():
        raise IsADirectoryError(f"输出 CSV 不能是目录: {args.output_csv}")

    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"图像目录不存在: {args.image_dir}")
    if not args.mask_dir.is_dir():
        raise NotADirectoryError(f"mask 目录不存在: {args.mask_dir}")

    image_files = collect_files(args.image_dir, IMAGE_EXTENSIONS)
    mask_files = collect_files(args.mask_dir, MASK_EXTENSIONS)

    rows: list[tuple[str, int]] = []
    missing_masks: list[str] = []

    for stem, image_path in sorted(image_files.items()):
        mask_path = mask_files.get(stem)
        if mask_path is None:
            missing_masks.append(image_path.name)
            continue

        foreground_pixels = count_foreground_pixels(mask_path)
        if foreground_pixels < args.threshold:
            rows.append((mask_path.name, foreground_pixels))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mask_filename", "foreground_pixel_count"])
        writer.writerows(rows)

    print(f"图像总数: {len(image_files)}")
    print(f"mask 总数: {len(mask_files)}")
    print(f"前景像素 < {args.threshold} 的 mask 数量: {len(rows)}")
    print(f"CSV 已保存到: {args.output_csv}")

    if missing_masks:
        print(f"未找到对应 mask 的图像数: {len(missing_masks)}")
        print("示例未匹配文件:")
        for name in missing_masks[:10]:
            print(f"  {name}")


if __name__ == "__main__":
    main()
