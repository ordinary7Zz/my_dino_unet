

# DINO U-Net

一个基于 DINO 和 U-Net 结构的图像分割模型代码仓库。

👤 **Code Author**: XinYu

##  文件说明
- **dino_unet.py**: 模型核心结构定义。
- **dataset.py**: 数据集加载与预处理。
- **train.py / train.sh**: 模型训练脚本。
- **eval.py / eval.sh**: 模型评估脚本。
- **test.py / test.sh**: 模型测试脚本。


## 安装依赖
```bash
pip install -r requirements.txt
```

## 淋巴结区域图像筛选

当前仓库已经支持基于分割模型的淋巴结区域图像筛选，核心调用链路如下：

1. `infer_recursive.py`
   - 对候选超声图像做递归推理；
   - 一次推理同时输出：
     - 二值掩码 `binary/*.png`
     - 概率图 `prob/*.npy`
2. `screen_lymph_node_region.py`
   - 读取原图、二值掩码和概率图；
   - 提取掩码形态特征和概率置信特征；
   - 输出 `yes / suspicious / no` 三分类结果与筛选报告。
3. `scripts/baseline/predict_Lymph_Node_Region_Screening.sh`
   - 将上述两步串联成一键执行脚本。

### 相关文件
- `infer_recursive.py`：递归推理并保存筛选所需的 mask / prob 结果
- `screen_lymph_node_region.py`：图像级筛选主脚本
- `configs/lymph_node_region_screening.yaml`：筛选阈值和策略配置
- `scripts/baseline/predict_Lymph_Node_Region_Screening.sh`：一键执行入口

### 方法一：直接使用一键脚本

先修改脚本顶部配置：
- `CHECKPOINT_PATH`：分割模型权重路径
- `INPUT_DIR`：待筛选图像目录
- `OUTPUT_ROOT`：筛选输出目录
- `CONFIG_PATH`：筛选配置文件路径
- `DEVICE` / `CUDA_VISIBLE_DEVICES`：推理设备

然后执行：

```bash
bash scripts/baseline/predict_Lymph_Node_Region_Screening.sh
```

该脚本会依次调用：

```bash
python -u infer_recursive.py ...
python -u screen_lymph_node_region.py ...
```

### 方法二：分两步手动执行

#### 第一步：生成二值掩码和概率图

```bash
python -u infer_recursive.py \
  --checkpoint "/path/to/checkpoint.pth" \
  --input_dir "/path/to/images" \
  --output_dir "/path/to/screening_outputs/masks" \
  --img_size 224 \
  --batch_size 4 \
  --num_workers 4 \
  --device "cuda:0" \
  --dino_pretrained "True" \
  --use_dilation "False" \
  --save_orig_size "True" \
  --threshold 0.5 \
  --save_binary "True" \
  --save_prob_npy "True"
```

执行后，输出目录下会生成：

```text
screening_outputs/
  masks/
    binary/
      xxx.png
    prob/
      xxx.npy
```

说明：
- `binary/*.png` 是阈值化后的前景掩码；
- `prob/*.npy` 是真实 sigmoid 概率图，供后续筛选脚本计算概率特征；
- 如果还想额外保存可视化概率图，可增加：

```bash
  --save_prob_png "True"
```

#### 第二步：执行图像级筛选

```bash
python -u screen_lymph_node_region.py \
  --image_dir "/path/to/images" \
  --binary_mask_dir "/path/to/screening_outputs/masks/binary" \
  --prob_mask_dir "/path/to/screening_outputs/masks/prob" \
  --output_csv "/path/to/screening_outputs/reports/screening_results.csv" \
  --config "configs/lymph_node_region_screening.yaml" \
  --save_subset_manifest "True" \
  --manifest_dir "/path/to/screening_outputs/reports"
```

### 配置文件说明

筛选规则由 `configs/lymph_node_region_screening.yaml` 控制，主要包括：
- `min_region_area`：最小主连通域面积
- `probability.high_conf_threshold`：高置信像素阈值
- `probability.uncertainty_lower / uncertainty_upper`：不确定区间
- `thresholds.*`：可信掩码规则阈值
- `negative_rules.*`：强负向规则阈值
- `label_thresholds.*`：`yes / suspicious / no` 分类阈值

如果需要调整筛选严格程度，优先修改该 YAML 文件，而不是直接改 Python 代码。

### 输出结果说明

筛选完成后，通常会得到如下目录结构：

```text
screening_outputs/
  masks/
    binary/
    prob/
    prob_png/         # 仅在 --save_prob_png=True 时生成
  reports/
    screening_results.csv
    yes_manifest.txt
    suspicious_manifest.txt
    no_manifest.txt
```

其中：
- `screening_results.csv`：每张图像的完整筛选结果；
- `present_pred`：最终分类，取值为 `yes / suspicious / no`；
- `confidence`：结果置信度；
- `final_score`：连续筛选分数；
- `hard_reject_flag`：是否触发强拒绝规则；
- `reason`：分类原因；
- `yes_manifest.txt / suspicious_manifest.txt / no_manifest.txt`：每类图像路径清单，便于人工复核或后续过滤。

### 典型使用流程

1. 准备待筛选超声图像目录；
2. 选择训练好的分割模型 checkpoint；
3. 运行 `infer_recursive.py` 生成 `binary` 和 `prob`；
4. 运行 `screen_lymph_node_region.py` 生成 CSV 和 manifest；
5. 优先保留 `yes`；
6. 对 `suspicious` 做人工复核；
7. 将 `no` 作为排除样本或进一步抽查。

## 使用 `filter_json_by_exported_paths.py` 过滤标注 JSON

当 `patient_summary.csv` 中的 `exported_relative_paths` 记录了筛选后导出的图像路径时，可以用这个脚本从原始标注 JSON 中只保留这些已导出的图像。

### 输入说明
- `--patient_summary_csv`：`infer_recursive_screening.py` 生成的 `patient_summary.csv`
- `--input_json`：待过滤的标注 JSON，顶层必须是列表，每条记录需要有 `filename` 字段
- `--output_json`：可选，输出路径；不传时默认输出为 `<input_stem>_exported_only.json`

### 使用示例
如果需要指定输出文件：

```bash
python utils/filter_json_by_exported_paths.py \
  --patient_summary_csv screening_inference_outputs/train_dataset/Malignant_ultrasound_images_cropped_screening/tables/patient_summary.csv \
  --input_json my_json/train_labels.json \
  --output_json my_json/train_labels_sample_exported_only.json
```

```bash
python utils/filter_json_by_exported_paths.py \
  --patient_summary_csv screening_inference_outputs/test_dataset/Malignant_ultrasound_images_cropped_screening/tables/patient_summary.csv \
  --input_json my_json/test_labels.json \
  --output_json my_json/test_labels_sample_exported_only.json
```

### 路径匹配说明
`patient_summary.csv` 中的 `exported_relative_paths` 使用 `|` 分隔多个图像路径，例如：

```text
刘惠银/刘惠银_01_0008_0008.jpg|刘惠银/刘惠银_01_0006_0006.jpg|刘惠银/刘惠银_01_0001_0001.jpg
```

脚本会自动按 `|` 拆分，并兼容以下两种路径格式的匹配：
- `患者名/图像.jpg`
- `2016/患者名/图像.jpg`

因此可以直接用于带年份目录前缀的 `filename` 字段。

### 运行结果
脚本运行后会打印：
- 读取到的导出路径数量
- 输入 JSON 记录数
- 保留记录数
- 删除记录数
- 输出 JSON 路径