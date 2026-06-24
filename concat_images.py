import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def find_corresponding_files(orig_dir, gt_dir, shap_dir):
    """根据文件名（不含后缀）匹配三个目录中的图像，返回 (stem, orig_path, gt_path, shap_path) 列表。"""
    orig_dir = Path(orig_dir)
    gt_dir = Path(gt_dir)
    shap_dir = Path(shap_dir)

    # 收集各目录的文件 stem -> [path]
    def build_stem_map(d):
        if not d.is_dir():
            raise NotADirectoryError(f"目录不存在: {d}")
        stem_map = {}
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}:
                stem_map[f.stem] = str(f)
        return stem_map

    orig_map = build_stem_map(orig_dir)
    gt_map = build_stem_map(gt_dir)
    shap_map = build_stem_map(shap_dir)

    # 找出三个目录共有的 stem
    common_stems = set(orig_map.keys()) & set(gt_map.keys()) & set(shap_map.keys())

    matched = [(s, orig_map[s], gt_map[s], shap_map[s]) for s in sorted(common_stems)]

    return matched


def add_label_to_image(img: np.ndarray, label: str, font_scale: float = 1.0, thickness: int = 2):
    """在图像顶部添加居中文字标注，返回带标签的新图像。"""
    h, w = img.shape[:2]
    # 计算文字所需的顶部边距
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_height = text_h + baseline + 12  # 上下留一点间距

    # 创建新画布：原图高度 + 标签高度
    new_img = np.ones((h + label_height, w, 3), dtype=np.uint8) * 255
    new_img[label_height:, :] = img

    # 居中绘制文字
    text_x = (w - text_w) // 2
    text_y = label_height - baseline - 4
    cv2.putText(
        new_img, label, (text_x, text_y),
        font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA,
    )
    return new_img


def resize_to_same_height(images: list, target_height: int, interpolation=cv2.INTER_LINEAR):
    """将列表中的图像统一缩放到同一高度（保持宽高比）。"""
    resized = []
    for img, label in images:
        h, w = img.shape[:2]
        new_w = int(w * target_height / h)
        resized.append((cv2.resize(img, (new_w, target_height), interpolation=interpolation), label))
    return resized


def concat_three(img_orig, img_gt, img_shap, labels):
    """水平拼接三张带标注的图像，返回拼接结果。"""
    # 统一到同一高度（取三张图中的最大高度）
    images_with_labels = [(img_orig, labels[0]), (img_gt, labels[1]), (img_shap, labels[2])]
    max_h = max(img.shape[0] for img, _ in images_with_labels)
    images_with_labels = resize_to_same_height(images_with_labels, max_h)

    # 给每张图添加顶部标注
    labeled_images = [add_label_to_image(img, lbl) for img, lbl in images_with_labels]

    # 水平拼接
    return np.hstack(labeled_images)


def main():
    parser = argparse.ArgumentParser(description="水平拼接原图、GT和SHAP图，并添加标注")
    parser.add_argument("--orig_dir", required=True, help="原图目录")
    parser.add_argument("--gt_dir", required=True, help="Ground Truth (origin_gt) 目录")
    parser.add_argument("--shap_dir", required=True, help="SHAP分析图目录")
    parser.add_argument("--output_dir", required=True, help="拼接结果输出目录")
    parser.add_argument("--label1", default="Original", help="第一张图的标注文字")
    parser.add_argument("--label2", default="Ground Truth", help="第二张图的标注文字")
    parser.add_argument("--label3", default="SHAP", help="第三张图的标注文字")
    args = parser.parse_args()

    # 匹配文件
    matched = find_corresponding_files(args.orig_dir, args.gt_dir, args.shap_dir)
    print(f"共匹配到 {len(matched)} 组图像")

    if len(matched) == 0:
        print("警告：没有找到三目录共有的文件")
        return

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = (args.label1, args.label2, args.label3)

    for stem, orig_path, gt_path, shap_path in matched:
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

        # 拼接
        result = concat_three(img_orig, img_gt, img_shap, labels)

        # 保存
        out_path = output_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), result)
        print(f"已保存: {out_path}")

    print("完成！")


if __name__ == "__main__":
    main()
