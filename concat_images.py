import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_corresponding_files(orig_dir, gt_dir, shap_dir):
    """根据文件名（不含后缀）匹配三个目录中的图像，返回 (stem, orig_path, gt_path, shap_path) 列表。"""
    orig_dir = Path(orig_dir)
    gt_dir = Path(gt_dir)
    shap_dir = Path(shap_dir)

    def build_stem_map(d):
        if not d.is_dir():
            raise NotADirectoryError(f"目录不存在: {d}")
        stem_map = {}
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
                stem_map[f.stem] = str(f)
        return stem_map

    orig_map = build_stem_map(orig_dir)
    gt_map = build_stem_map(gt_dir)
    shap_map = build_stem_map(shap_dir)

    common_stems = set(orig_map.keys()) & set(gt_map.keys()) & set(shap_map.keys())
    matched = [(s, orig_map[s], gt_map[s], shap_map[s]) for s in sorted(common_stems)]
    return matched


def main():
    parser = argparse.ArgumentParser(description="水平拼接原图、GT和SHAP图，并添加标注（使用 matplotlib）")
    parser.add_argument("--orig_dir", required=True, help="原图目录")
    parser.add_argument("--gt_dir", required=True, help="Ground Truth (origin_gt) 目录")
    parser.add_argument("--shap_dir", required=True, help="SHAP分析图目录")
    parser.add_argument("--output_dir", required=True, help="拼接结果输出目录")
    parser.add_argument("--label1", default="Original", help="第一张图的标注文字")
    parser.add_argument("--label2", default="Ground Truth", help="第二张图的标注文字")
    parser.add_argument("--label3", default="SHAP", help="第三张图的标注文字")
    parser.add_argument("--dpi", type=int, default=150, help="输出图像的 DPI（默认 150）")
    parser.add_argument("--fontsize", type=int, default=24, help="标注文字字号（默认 24）")
    args = parser.parse_args()

    matched = find_corresponding_files(args.orig_dir, args.gt_dir, args.shap_dir)
    print(f"共匹配到 {len(matched)} 组图像")

    if len(matched) == 0:
        print("警告：没有找到三目录共有的文件")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = (args.label1, args.label2, args.label3)

    for stem, orig_path, gt_path, shap_path in matched:
        # 读取三张图（cv2 读入是 BGR，转 RGB）
        img_orig = cv2.imread(orig_path)
        img_gt = cv2.imread(gt_path)
        img_shap = cv2.imread(shap_path)

        if img_orig is None:
            print(f"跳过 {stem}：无法读取原图 {orig_path}")
            continue
        if img_gt is None:
            print(f"跳过 {stem}：无法读取GT图 {gt_path}")
            continue
        if img_shap is None:
            print(f"跳过 {stem}：无法读取SHAP图 {shap_path}")
            continue

        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_shap = cv2.cvtColor(img_shap, cv2.COLOR_BGR2RGB)

        # 统一到最大高度（保持宽高比）
        images = [img_orig, img_gt, img_shap]
        max_h = max(img.shape[0] for img in images)
        resized = []
        for img in images:
            h, w = img.shape[:2]
            new_w = int(w * max_h / h)
            resized.append(cv2.resize(img, (new_w, max_h)))

        widths = [img.shape[1] for img in resized]
        total_w = sum(widths)

        # 用 subplot 创建水平拼接图，width_ratios 保持每张图的真实宽高比
        fig, axes = plt.subplots(
            1, 3,
            figsize=(total_w / args.dpi, max_h / args.dpi),
            gridspec_kw={"width_ratios": widths, "wspace": 0.02},
        )
        for ax, img, label in zip(axes, resized, labels):
            ax.imshow(img)
            ax.set_title(label, fontsize=args.fontsize, fontweight="normal", pad=6)
            ax.axis("off")

        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)

        out_path = output_dir / f"{stem}.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"已保存: {out_path}")

    print("完成！")


if __name__ == "__main__":
    main()
