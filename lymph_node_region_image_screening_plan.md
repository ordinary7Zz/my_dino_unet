# 基于分割模型辅助的淋巴结区域图像筛选方案

## 1. 背景与目标

当前仓库中的分割模型在淋巴结区域超声图像数据集上训练，训练脚本见 `scripts/baseline/train_Lymph_Node_Metastasis.sh`，训练主流程见 `train.py`，数据读取逻辑见 `dataset.py`。

该模型学习到的是：

- 在淋巴结区域超声图像中，目标区域通常如何出现；
- 目标区域的边界、面积分布和局部形态特征；
- 输入图像与目标掩码之间的映射关系。

因此，可以将该分割模型作为“是否包含淋巴结区域”的辅助工具，用于增强图像筛选过程中的可信度判断。

本方案的目标不是让分割模型直接替代人工判读，而是构建一套**分割辅助筛选流程**，将候选图像分为三类：

- `yes`：明确包含淋巴结区域；
- `suspicious`：疑似包含淋巴结区域，需要人工复核；
- `no`：不包含淋巴结区域。

---

## 2. 核心思想

核心思想如下：

1. 先使用现有分割模型对候选超声图像进行推理；
2. 同时保留：
   - 二值掩码（binary mask）
   - 概率图（probability map）
3. 从掩码与概率图中提取图像级特征；
4. 根据“掩码是否像一个合理的淋巴结区域”以及“模型是否对该预测有足够置信度”进行打分；
5. 输出图像级筛选结果和辅助解释信息。

需要强调的是：

- **分割模型适合作为辅助证据，而不是唯一判据**；
- **更适合用于负向排除与置信度增强，不适合直接替代人工规则**；
- **异常淋巴结或分布外图像可能导致分割结果异常，因此应保留 `suspicious` 档供人工复核**。

---

## 3. 筛选流程设计

### 3.1 总体流程

```text
候选超声图像目录
    ↓
分割模型批量推理
    ↓
输出 binary mask + probability map
    ↓
提取掩码形态特征 + 概率置信特征
    ↓
规则打分与分类
    ↓
输出筛选结果 CSV
    ↓
人工复核 suspicious / low-confidence 样本
```

### 3.2 分类目标

每张图像最终输出以下字段：

- `present_pred`: `yes / suspicious / no`
- `confidence`: `high / medium / low`
- `score`: 连续分数，用于排序和阈值调优
- `reason`: 判定原因描述
- `hard_reject_flag`: 是否触发强拒绝规则

---

## 4. 分割模型在筛选中的作用边界

### 4.1 适合利用的能力

分割模型可用于判断：

- 是否存在一个相对稳定的前景区域；
- 该前景区域是否具备单一主连通域；
- 该前景区域形态上是否接近椭圆/豆形；
- 模型对预测区域是否具备相对高的概率响应；
- 预测是否过于碎片化、不稳定或接近空掩码。

### 4.2 不适合直接承担的任务

分割模型不能直接等价于图像级存在性分类器，原因包括：

1. 训练任务是像素级分割，不是图像级 yes/no 分类；
2. 模型可能对非淋巴结结构产生误分割；
3. 异常淋巴结可能偏离典型形态，导致掩码不规则；
4. 未标注样本上无法直接用 Dice/ECE 等监督指标评价单图结果。

因此，系统设计必须保留：

- 人工规则主导；
- 分割结果辅助；
- 中间状态 `suspicious`；
- 可解释的分数与原因输出。

---

## 5. 特征设计

筛选特征分为两大类：

1. 掩码形态特征
2. 概率置信特征

这些特征均在单张图像级别计算。

### 5.1 掩码形态特征

用于回答：**分出来的区域是否像一个淋巴结区域**。

建议提取以下特征：

#### 5.1.1 前景面积相关

- `area_ratio`
  - 定义：前景像素数 / 全图像素数
  - 作用：空掩码、极小掩码、极大掩码都可能表示异常

- `largest_component_area_ratio`
  - 定义：最大连通域面积 / 全图像素数
  - 作用：判断主目标是否足够显著

- `largest_component_ratio`
  - 定义：最大连通域面积 / 全部前景面积
  - 作用：判断预测是否由单一主体构成

#### 5.1.2 连通域相关

- `component_count`
  - 定义：前景连通域数量
  - 作用：碎片化掩码通常提示不稳定或非目标结构

- `small_component_ratio`
  - 定义：小连通域总面积 / 全部前景面积
  - 作用：碎片噪声比例越高，可信度越低

#### 5.1.3 外接框与形状相关

- `bbox_width`
- `bbox_height`
- `bbox_aspect_ratio`
  - 定义：外接框长边 / 短边
  - 作用：用于判断是否接近椭圆/豆形，而不是管状、片状

- `extent`
  - 定义：前景面积 / 外接框面积
  - 作用：过低可能提示结构分散、边界异常

- `perimeter`
  - 定义：最大连通域周长

- `compactness`
  - 定义：`perimeter^2 / (4π * area)`
  - 作用：越高说明边界越不规则

- `eccentricity`
  - 定义：主轴与次轴差异程度
  - 作用：辅助判断是否具备椭圆状结构

- `solidity`
  - 定义：面积 / 凸包面积
  - 作用：越低通常边缘越毛糙或凹陷越明显

#### 5.1.4 边界接触相关

- `touch_border_top`
- `touch_border_bottom`
- `touch_border_left`
- `touch_border_right`
- `touch_border_count`
- `touch_border_ratio`

作用：

- 如果大面积掩码贴边，可能代表截断结构、背景误分割或非标准 ROI；
- 对于明显压边的大掩码，可以降低可信度。

#### 5.1.5 内部结构相关

- `hole_count`
- `hole_ratio`

作用：

- 正常单目标区域若内部空洞过多，可能代表不稳定预测；
- 但该特征只作为辅助手段，避免过度依赖。

---

### 5.2 概率置信特征

用于回答：**模型是否对当前预测区域有足够把握**。

> 注意：这里不能直接使用训练/验证中使用的 ECE，因为 ECE 需要真实标签；未标注筛选阶段只能依赖无监督置信特征。

建议提取以下特征：

#### 5.2.1 全局概率特征

- `global_mean_prob`
  - 定义：整张图像概率均值

- `global_max_prob`
  - 定义：整张图像概率最大值

- `global_std_prob`
  - 定义：整张图像概率标准差

作用：

- 若 `global_max_prob` 很低，说明模型整体没有明确激活区域；
- 若概率分布过于平坦，通常表示模型犹豫。

#### 5.2.2 前景区域概率特征

基于二值掩码前景区域计算：

- `fg_mean_prob`
- `fg_median_prob`
- `fg_max_prob`
- `fg_std_prob`

作用：

- 前景平均概率高，通常表示模型对该区域较有把握；
- 前景波动过大，可能表示边界不稳定。

#### 5.2.3 高置信与不确定像素比例

- `high_conf_fg_ratio`
  - 定义：前景区域中 `prob >= high_conf_threshold` 的像素比例

- `uncertainty_ratio`
  - 定义：全图中 `prob` 落在 `[0.4, 0.6]` 或配置区间内的像素比例

- `mid_conf_ratio`
  - 定义：全图中中等置信度区域比例

作用：

- 高置信前景比例越高越支持正样本；
- 不确定像素比例过大说明模型判断不清晰。

#### 5.2.4 熵特征

- `entropy_mean`
- `entropy_fg_mean`

单像素熵定义：

```text
H(p) = -p*log(p) - (1-p)*log(1-p)
```

作用：

- 熵高表示模型不确定性高；
- 可作为未标注场景下的辅助置信信号。

---

## 6. 判定规则设计

不建议使用单一阈值直接判断，建议采用：

1. 强拒绝规则（hard reject）
2. 支持规则（support rules）
3. 综合得分（score）
4. 最终分类映射（yes / suspicious / no）

### 6.1 强拒绝规则

若满足以下情况之一，可直接标记 `hard_reject_flag = true`，并优先判定为 `no` 或 `suspicious`：

- `area_ratio` 小于最小阈值，掩码近似为空；
- `global_max_prob` 明显过低；
- `component_count` 过高且 `largest_component_ratio` 过低；
- `uncertainty_ratio` 过高；
- 掩码面积异常大且大面积贴边；
- 主连通域形态明显细长或极不规则。

### 6.2 支持规则

满足以下多项时，增加 `yes` 倾向：

- 存在单一主连通域；
- 主连通域面积在合理范围内；
- `bbox_aspect_ratio` 落在合理区间；
- `solidity` 较高；
- `compactness` 适中；
- `fg_mean_prob` 较高；
- `high_conf_fg_ratio` 较高；
- `touch_border_ratio` 较低。

### 6.3 综合得分

建议将所有支持/惩罚项映射为 0~1 或 -1~1，再形成总分：

```text
final_score = shape_score * w_shape + prob_score * w_prob + penalty_score * w_penalty
```

其中：

- `shape_score`：来自形态特征
- `prob_score`：来自概率与不确定性特征
- `penalty_score`：来自强负向特征
- `w_shape / w_prob / w_penalty`：配置文件中定义的权重

### 6.4 分类映射

建议：

- `final_score >= yes_threshold` → `yes`
- `suspicious_threshold <= final_score < yes_threshold` → `suspicious`
- `final_score < suspicious_threshold` → `no`

若触发 `hard_reject_flag`：

- 可直接降为 `no`
- 或将最高分类上限限制为 `suspicious`

具体策略建议在配置文件中可调。

---

## 7. 代码建设方案

本方案建议采用“尽量复用现有推理逻辑 + 新增筛选模块”的方式实现。

### 7.1 需要改造的现有代码

#### 7.1.1 `infer_recursive.py`

**作用**：批量推理候选图像，输出二值掩码与概率图。

**当前现状**：

- 已支持递归读取目录；
- 已支持推理；
- 已支持 `binary` 或 `prob` 单一输出模式。

**建议改造内容**：

新增同时输出两种结果的能力，例如：

- `--output_type both`

或改为两个独立开关：

- `--save_binary true`
- `--save_prob true`

**目标输出结构示例**：

```text
output_dir/
  binary/
    xxx.png
  prob/
    xxx.png
```

**改造原因**：

- 避免为同一批图运行两次推理；
- 确保二值掩码与概率图严格对应；
- 方便后续批量筛选脚本直接读取。

---

### 7.2 需要新增的代码文件

#### 7.2.1 `utils/mask_region_features.py`

**作用**：从二值掩码与概率图中提取图像级筛选特征。

**输入**：

- 二值掩码路径或数组
- 概率图路径或数组
- 可选：原图尺寸

**输出**：

- Python dict，包含所有形态与概率特征

**建议函数设计**：

- `load_binary_mask(path)`
- `load_prob_map(path)`
- `get_connected_components(mask)`
- `select_largest_component(mask)`
- `compute_shape_features(mask)`
- `compute_prob_features(prob, mask)`
- `compute_entropy_features(prob, mask)`
- `extract_region_features(binary_mask, prob_map)`

**说明**：

该模块应尽量独立、纯函数化，方便后续复用到：

- 规则筛选
- 阈值标定
- 数据分析
- 轻量分类器训练

---

#### 7.2.2 `screen_lymph_node_region.py`

**作用**：读取推理结果，完成特征提取、规则打分、最终分类，并输出结果表。

这是整个方案的核心主脚本。

**输入参数建议**：

- `--image_dir`
- `--binary_mask_dir`
- `--prob_mask_dir`
- `--output_csv`
- `--config`
- `--save_subset_manifest`
- `--copy_mode`（可选，先支持 `none/manifest` 即可）

**建议主流程**：

1. 遍历候选图像；
2. 查找对应 binary mask 与 prob map；
3. 调用 `utils/mask_region_features.py` 提取特征；
4. 依据配置文件完成规则打分；
5. 生成 `yes/suspicious/no` 结果；
6. 输出 CSV。

**建议函数设计**：

- `load_config(config_path)`
- `score_shape_features(features, cfg)`
- `score_prob_features(features, cfg)`
- `check_hard_reject(features, cfg)`
- `build_reason(features, cfg)`
- `classify_score(score, hard_reject, cfg)`
- `process_one_case(image_path, binary_path, prob_path, cfg)`
- `write_results_csv(rows, output_csv)`
- `main()`

**输出内容建议**：

基础字段：

- `image_path`
- `binary_mask_path`
- `prob_mask_path`
- `present_pred`
- `confidence`
- `final_score`
- `hard_reject_flag`
- `reason`

附加字段：

- 所有核心形态特征
- 所有核心概率特征

---

#### 7.2.3 `configs/lymph_node_region_screening.yaml`

**作用**：统一配置阈值、权重和分类策略。

**为什么必须配置化**：

- 各数据集分布不同，阈值会调整；
- 医学图像筛选规则往往需要多轮校准；
- 写死到代码中不利于复现实验。

**建议配置内容结构**：

```yaml
mask_threshold: 0.5

probability:
  high_conf_threshold: 0.8
  uncertainty_lower: 0.4
  uncertainty_upper: 0.6

hard_reject:
  min_area_ratio: 0.003
  max_area_ratio: 0.7
  max_component_count: 6
  min_largest_component_ratio: 0.45
  min_global_max_prob: 0.55
  max_uncertainty_ratio: 0.30
  max_touch_border_ratio: 0.60

shape_rules:
  area_ratio_min: 0.01
  area_ratio_max: 0.35
  aspect_ratio_min: 1.2
  aspect_ratio_max: 4.5
  min_solidity: 0.75
  max_compactness: 3.5

prob_rules:
  min_fg_mean_prob: 0.65
  min_high_conf_fg_ratio: 0.50
  max_entropy_mean: 0.62

weights:
  shape: 0.45
  prob: 0.35
  penalty: 0.20

thresholds:
  yes: 0.70
  suspicious: 0.45

policy:
  hard_reject_to_no: true
```

注意：以上数值只是首版默认值，最终应通过人工标注样本校准。

---

#### 7.2.4 `calibrate_region_screening_thresholds.py`

**作用**：基于少量人工标注样本对筛选规则进行阈值分析与参数校准。

**定位**：

- 不是筛选主链路必需文件；
- 但对于让规则从“经验阈值”变为“数据驱动阈值”非常关键。

**输入**：

- 小规模人工标注表，例如：
  - `image_path`
  - `gt_present`（yes / suspicious / no）
- 推理结果目录或筛选特征 CSV

**输出**：

- 特征分布统计
- 推荐阈值
- 分类结果分析
- 可选：ROC/PR 结果
- 阈值建议报告

**建议函数设计**：

- `load_labeled_cases()`
- `merge_features_with_labels()`
- `analyze_feature_distribution()`
- `search_best_thresholds()`
- `evaluate_rule_set()`
- `export_recommendations()`

---

#### 7.2.5 `scripts/baseline/predict_Lymph_Node_Region_Screening.sh`

**作用**：将整个流程串联成一键执行脚本。

**建议流程**：

1. 调用 `infer_recursive.py` 进行批量推理；
2. 同时输出 binary mask 和 prob map；
3. 调用 `screen_lymph_node_region.py` 做图像级筛选；
4. 输出最终 CSV 与清单文件。

**建议输入参数**：

- checkpoint 路径
- 输入图像目录
- 输出目录
- config 路径
- device

**作用价值**：

- 与现有 `scripts/baseline/*.sh` 风格保持一致；
- 便于复现、批处理与后续自动化。

---

## 8. 文件级职责划分

建议最终的代码组织如下：

```text
infer_recursive.py
screen_lymph_node_region.py
calibrate_region_screening_thresholds.py
configs/
  lymph_node_region_screening.yaml
utils/
  mask_region_features.py
scripts/
  baseline/
    predict_Lymph_Node_Region_Screening.sh
```

各文件职责：

- `infer_recursive.py`
  - 负责模型推理与保存预测结果
- `utils/mask_region_features.py`
  - 负责特征计算
- `screen_lymph_node_region.py`
  - 负责筛选主逻辑
- `configs/lymph_node_region_screening.yaml`
  - 负责参数配置
- `calibrate_region_screening_thresholds.py`
  - 负责阈值校准与分析
- `scripts/baseline/predict_Lymph_Node_Region_Screening.sh`
  - 负责流程编排

---

## 9. 数据输出设计

### 9.1 推理输出目录

建议输出结构：

```text
screening_outputs/
  masks/
    binary/
      case1.png
      case2.png
    prob/
      case1.png
      case2.png
  reports/
    screening_results.csv
    yes_manifest.txt
    suspicious_manifest.txt
    no_manifest.txt
```

### 9.2 `screening_results.csv` 字段建议

#### 基础字段

- `image_path`
- `binary_mask_path`
- `prob_mask_path`
- `present_pred`
- `confidence`
- `final_score`
- `hard_reject_flag`
- `reason`

#### 形态特征字段

- `area_ratio`
- `largest_component_area_ratio`
- `largest_component_ratio`
- `component_count`
- `small_component_ratio`
- `bbox_aspect_ratio`
- `extent`
- `compactness`
- `eccentricity`
- `solidity`
- `touch_border_ratio`
- `hole_ratio`

#### 概率特征字段

- `global_mean_prob`
- `global_max_prob`
- `global_std_prob`
- `fg_mean_prob`
- `fg_median_prob`
- `fg_max_prob`
- `fg_std_prob`
- `high_conf_fg_ratio`
- `uncertainty_ratio`
- `entropy_mean`
- `entropy_fg_mean`

### 9.3 清单文件

建议额外输出：

- `yes_manifest.txt`
- `suspicious_manifest.txt`
- `no_manifest.txt`

每行一个原图路径，便于：

- 人工复核
- 数据复制/链接
- 后续训练前过滤

---

## 10. 与当前仓库现有代码的关系

### 10.1 可直接复用的部分

- `infer_recursive.py`
  - 已有递归推理能力
- `train.py`
  - 可用于理解当前模型输入输出格式
- `dataset.py`
  - 可作为图像预处理参考
- `utils/metrics.py`
  - 可参考 ECE/概率相关处理方式，但不直接用于未标注筛选

### 10.2 不建议直接改动的部分

不建议将图像筛选逻辑直接塞入：

- `train.py`
- `dataset.py`

原因：

1. 筛选属于训练前的离线预处理；
2. 若混入训练主流程，会增加耦合；
3. 规则迭代频繁，不适合与训练循环绑定。

因此建议保持：

- 训练代码独立；
- 筛选代码独立；
- 两者通过筛选后的图像清单衔接。

---

## 11. 推荐的实现顺序

### 第一阶段：建立可运行版本

1. 改造 `infer_recursive.py`，支持同时输出 binary 和 prob；
2. 新增 `utils/mask_region_features.py`；
3. 新增 `screen_lymph_node_region.py`；
4. 新增 `configs/lymph_node_region_screening.yaml`；
5. 新增 `scripts/baseline/predict_Lymph_Node_Region_Screening.sh`。

该阶段完成后，可以跑通整个筛选流程并输出首版结果。

### 第二阶段：校准阈值

6. 准备少量人工标注数据；
7. 新增 `calibrate_region_screening_thresholds.py`；
8. 基于验证结果更新 YAML 阈值配置。

该阶段完成后，可显著提升规则稳定性与可信度。

### 第三阶段：进一步增强

若规则筛选效果有限，可进一步考虑：

- 使用这些特征训练轻量图像级分类器；
- 采用多任务模型，同时输出分割结果与是否包含淋巴结区域；
- 增加可视化报告，输出原图 + mask + prob + decision overlay。

但这些属于后续增强，不建议在第一版中一起实现。

---

## 12. 风险与注意事项

### 12.1 异常淋巴结风险

异常或恶性淋巴结可能：

- 形态不典型；
- 边界不规则；
- 脂肪门消失；
- 坏死、钙化、回声不均。

因此，若只依据“是否像典型椭圆结构”进行筛选，容易误杀真实异常病例。为避免此问题：

- 不要只用单一形状阈值；
- 保留 `suspicious`；
- 对异常形态样本优先人工复核。

### 12.2 分布外图像风险

对于明显不在训练分布内的图像，模型可能：

- 预测空掩码；
- 随机分出噪声区域；
- 输出不稳定概率图。

因此应同时依赖：

- 形态特征；
- 概率特征；
- 强拒绝规则。

### 12.3 阈值泛化风险

不同中心、不同设备、不同预处理方式下，特征分布可能不同。建议：

- 阈值配置化；
- 分中心校准；
- 保存每次筛选报告，便于回溯分析。

---

## 13. 最终结论

这套方案的本质是：

> 用现有分割模型提供“结构存在性 + 掩码合理性 + 概率置信度”三类辅助信号，构建一套训练前的图像筛选系统。

它的优势在于：

- 复用现有模型与推理代码；
- 不改训练主链路；
- 输出可解释；
- 易于逐步迭代；
- 适合作为数据过滤和人工复核前的预筛步骤。

它的边界在于：

- 不能完全替代人工诊断规则；
- 不能直接当作图像级分类器；
- 必须通过 `suspicious` 和人工复核机制控制误筛风险。

因此，推荐按“**规则主导、分割辅助、人工兜底**”的原则实施。
