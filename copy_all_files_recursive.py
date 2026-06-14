"""
递归复制文件工具。

输入：
- --src_dir：要递归扫描的源目录
- --dst_dir：要复制到的目标目录

输出：
- 将源目录下的所有文件复制到目标目录。
- 复制前会统计文件总数。
- 复制时会把源目录下的相对路径用 "_" 拼接成新的文件名。
- 会先预览前 10 个重命名结果，确认后才开始实际复制。
"""

from pathlib import Path
import argparse
import shutil


def build_flat_filename(src_root: Path, file_path: Path) -> str:
    relative_path = file_path.relative_to(src_root)
    if len(relative_path.parts) == 1:
        return relative_path.name

    filename = relative_path.name
    if "." in filename:
        base_name = filename.rsplit(".", 1)[0]
        suffix = "." + filename.rsplit(".", 1)[1]
    else:
        base_name = filename
        suffix = ""

    flat_base = "_".join(list(relative_path.parts[:-1]) + [base_name])
    return flat_base + suffix


def copy_all_files(src_dir: str, dst_dir: str) -> None:
    src = Path(src_dir).resolve()
    dst = Path(dst_dir).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    file_paths = sorted((p for p in src.rglob("*") if p.is_file()), key=lambda p: str(p.relative_to(src)))
    total_files = len(file_paths)
    print(f"Total files to copy: {total_files}")

    print("Preview of first 10 renamed files:")
    for file_path in file_paths[:10]:
        target_name = build_flat_filename(src, file_path)
        print(f"  {file_path.relative_to(src)} -> {target_name}")

    if total_files > 10:
        print(f"  ... and {total_files - 10} more files")

    confirm = input("Proceed with copying? [y/N]: ").strip().lower()
    if confirm not in {"y", "yes"}:
        print("Copying cancelled by user.")
        return

    for file_path in file_paths:
        target_name = build_flat_filename(src, file_path)
        target_path = dst / target_name
        shutil.copy2(file_path, target_path)
        print(f"Copied: {file_path} -> {target_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursively copy all files from a source directory to a target directory.")
    parser.add_argument("--src_dir", required=True, help="Source directory")
    parser.add_argument("--dst_dir", required=True, help="Target directory")
    args = parser.parse_args()

    copy_all_files(args.src_dir, args.dst_dir)


if __name__ == "__main__":
    main()
