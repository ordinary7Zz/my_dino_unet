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

使用 matplotlib 将原图、Ground Truth 图和 SHAP 分析图水平拼接为一张图，并在每张子图上方添加标注。

```bash
python concat_images.py \
  --orig_dir "/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/pyradiomics_dice/gradcam/BM_171_doctor_wrong/original" \
  --gt_dir "/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/pyradiomics_dice/gradcam/BM_171_doctor_wrong/original_gt" \
  --shap_dir "/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/pyradiomics_train/binary_class/single_image/BM_171_doctor_wrong/compact_shap_bar" \
  --output_dir "/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_ori/pyradiomics_dice/concat_results" \
  --label1 "Original" \
  --label2 "Ground Truth" \
  --label3 "SHAP Analysis" \
  --dpi 150 \
  --fontsize 48
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--orig_dir` | (必填) | 原图目录 |
| `--gt_dir` | (必填) | Ground Truth (origin_gt) 目录 |
| `--shap_dir` | (必填) | SHAP 分析图目录 |
| `--output_dir` | (必填) | 拼接结果输出目录 |
| `--label1` | `Original` | 第1张图标注文字 |
| `--label2` | `Ground Truth` | 第2张图标注文字 |
| `--label3` | `SHAP` | 第3张图标注文字 |
| `--dpi` | `150` | 输出图像 DPI |
| `--fontsize` | `24` | 标注文字字号 |

三个目录中的图像按文件名 stem（不含后缀）自动匹配，统一缩放到最大高度后水平拼接，输出文件名为 `{stem}.png`。

## `infer_seg.py` 使用示例

二分类分割推理脚本，输入图像目录和模型权重，输出二值掩码 PNG（大小与原图一致）。

```bash
python infer_seg.py \
  --checkpoint ./checkpoints/baseline/dino_unet_train_dataset_4_epoch_50.pth \
  --input_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/sample/images \
  --output_dir ./my_infer/nodule \
  --img_size 224 \
  --batch_size 4 \
  --device cuda
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | (必填) | 模型权重路径 (.pth) |
| `--input_dir` | (必填) | 输入图像目录 |
| `--output_dir` | (必填) | 输出掩码目录 |
| `--img_size` | `224` | 模型输入图像尺寸 |
| `--batch_size` | `4` | 推理批大小 |
| `--num_workers` | `4` | DataLoader 进程数 |
| `--device` | `auto` | 推理设备 (cuda/cuda:0/cpu) |
| `--dino_pretrained` | `false` | 是否加载 DINO 预训练 backbone |
| `--use_dilation` | `false` | 模型是否使用 dilation 模块 |

输出：二值掩码 PNG（0/255），文件名与原图相同，尺寸还原至原始图像大小。
