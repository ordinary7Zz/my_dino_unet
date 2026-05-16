

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
  --patient_summary_csv screening_inference_outputs/2016_screening/tables/patient_summary.csv \
  --input_json my_json/train_labels.json \
  --output_json my_json/train_labels_sample_exported_only.json
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