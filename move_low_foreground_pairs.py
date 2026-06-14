import argparse
import csv
import shutil
from pathlib import Path


def collect_files_by_stem(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        stem = path.stem
        if stem in files:
            raise ValueError(f"发现重复图像文件名（去掉后缀后同名）：{stem}")
        files[stem] = path
    return files


def collect_files_by_name(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if name in files:
            raise ValueError(f"发现重复文件名：{name}")
        files[name] = path
    return files


def read_mask_filenames(csv_path: Path) -> list[str]:
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    last_error = None

    for encoding in encodings:
        try:
            with open(csv_path, "r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV 为空或缺少表头: {csv_path}")
                if "mask_filename" not in reader.fieldnames:
                    raise ValueError(f"CSV 必须包含 mask_filename 列: {csv_path}")

                mask_filenames = []
                for row in reader:
                    mask_filename = (row.get("mask_filename") or "").strip()
                    if mask_filename:
                        mask_filenames.append(mask_filename)
                return mask_filenames
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error if last_error else ValueError(f"无法读取 {csv_path}")


def move_pair(image_path: Path, mask_path: Path, image_dst_dir: Path, mask_dst_dir: Path) -> tuple[Path, Path]:
    image_dst = image_dst_dir / image_path.name
    mask_dst = mask_dst_dir / mask_path.name

    if image_dst.exists():
        raise FileExistsError(f"目标图像已存在: {image_dst}")
    if mask_dst.exists():
        raise FileExistsError(f"目标 mask 已存在: {mask_dst}")

    shutil.move(str(image_path), str(image_dst))
    shutil.move(str(mask_path), str(mask_dst))
    return image_dst, mask_dst


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 CSV 中的 mask_filename 移动对应的 image 和 mask。")
    parser.add_argument("csv_path", type=Path, help="stat_low_foreground_masks.py 输出的 CSV 文件")
    parser.add_argument("image_dir", type=Path, help="原始图像目录")
    parser.add_argument("mask_dir", type=Path, help="原始 mask 目录")
    parser.add_argument("output_dir", type=Path, help="新的输出目录")
    args = parser.parse_args()

    if not args.csv_path.is_file():
        raise FileNotFoundError(f"CSV 文件不存在: {args.csv_path}")
    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"图像目录不存在: {args.image_dir}")
    if not args.mask_dir.is_dir():
        raise NotADirectoryError(f"mask 目录不存在: {args.mask_dir}")

    mask_filenames = read_mask_filenames(args.csv_path)
    image_files = collect_files_by_stem(args.image_dir)
    mask_files = collect_files_by_name(args.mask_dir)

    image_dst_dir = args.output_dir / "images"
    mask_dst_dir = args.output_dir / "masks"
    image_dst_dir.mkdir(parents=True, exist_ok=True)
    mask_dst_dir.mkdir(parents=True, exist_ok=True)

    moved_count = 0
    missing_images: list[str] = []
    missing_masks: list[str] = []

    for mask_filename in mask_filenames:
        mask_path = mask_files.get(mask_filename)
        if mask_path is None:
            missing_masks.append(mask_filename)
            continue

        image_stem = Path(mask_filename).stem
        image_path = image_files.get(image_stem)
        if image_path is None:
            missing_images.append(mask_filename)
            continue

        move_pair(image_path, mask_path, image_dst_dir, mask_dst_dir)
        del image_files[image_stem]
        del mask_files[mask_filename]
        moved_count += 1

    print(f"CSV 中记录的 mask 数量: {len(mask_filenames)}")
    print(f"成功移动的图像/mask 对数量: {moved_count}")
    print(f"输出图像目录: {image_dst_dir}")
    print(f"输出 mask 目录: {mask_dst_dir}")

    if missing_masks:
        print(f"找不到对应 mask 的条目数: {len(missing_masks)}")
        for name in missing_masks[:10]:
            print(f"  缺失 mask: {name}")

    if missing_images:
        print(f"找不到对应图像的条目数: {len(missing_images)}")
        for name in missing_images[:10]:
            print(f"  缺失图像: {name}")


if __name__ == "__main__":
    main()
