## `stat_low_foreground_masks.py` 使用示例

```bash
python3 stat_low_foreground_masks.py \
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/VIDEO_ImageAndMask_merged/selected_images" \
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/VIDEO_ImageAndMask_merged/selected_masks" \
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/VIDEO_ImageAndMask_merged/mask_none.csv" \
  --threshold 10
```

## `move_low_foreground_pairs.py` 使用示例

这个脚本会根据 `stat_low_foreground_masks.py` 生成的 CSV，移动对应的图像和 mask 到新的输出目录。输出目录下会自动创建 `images/` 和 `masks/` 两个子目录。

```bash
python3 move_low_foreground_pairs.py \
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/VIDEO_ImageAndMask_merged/mask_none.csv" \
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/VIDEO_ImageAndMask_merged/selected_images" \
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/VIDEO_ImageAndMask_merged/selected_masks" \
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/VIDEO_ImageAndMask_merged/low_foreground_pairs"
```

参数顺序：

1. `csv_path`：`stat_low_foreground_masks.py` 输出的 CSV 文件，必须包含 `mask_filename` 列
2. `image_dir`：原始图像目录，脚本按文件名去掉后缀后的 `stem` 匹配 mask
3. `mask_dir`：原始 mask 目录，脚本按 `mask_filename` 精确匹配文件名
4. `output_dir`：移动后的输出目录

运行后，匹配到的图像会被移动到 `output_dir/images/`，对应的 mask 会被移动到 `output_dir/masks/`

## `concat_images.py` 使用示例

将原图、Ground Truth 图和 SHAP 分析图水平拼接为一张图，

```bash
python concat_images.py \
  --orig_dir "/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/pyradiomics_dice/gradcam/BM_171_doctor_wrong/original" \
  --gt_dir "/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/pyradiomics_dice/gradcam/BM_171_doctor_wrong/original_gt" \
  --shap_dir "/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/pyradiomics_train/binary_class/single_image/BM_171_doctor_wrong/compact_shap_bar" \
  --output_dir "/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/pyradiomics_dice/concat_results" \
  --label1 "Original" \
  --label2 "Ground Truth" \
  --label3 "SHAP Analysis"
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--orig_dir` | 原图目录 |
| `--gt_dir` | Ground Truth (origin_gt) 目录 |
| `--shap_dir` | SHAP 分析图目录 |
| `--output_dir` | 拼接结果输出目录 |
| `--label1` | 第1张图标注文字，默认 `Original` |
| `--label2` | 第2张图标注文字，默认 `Ground Truth` |
| `--label3` | 第3张图标注文字，默认 `SHAP` |

三个目录中的图像按文件名 stem（不含后缀）自动匹配，输出文件名为 `{stem}_concat.png`。
