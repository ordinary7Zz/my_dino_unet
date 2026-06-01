# 超声图像是否包含淋巴结区域的判断逻辑

## 目标
判断一张超声图像中是否存在**淋巴结区域**，并尽量区分“疑似淋巴结结构”和“非淋巴结结构/背景组织”。

## 核心思路
淋巴结在超声下通常不是只靠单一特征判断，而是结合：
1. **形态**：是否呈椭圆/豆形，长短轴比例是否偏大。
2. **内部结构**：是否存在**脂肪门（echogenic hilum）**，内部回声是否均匀。
3. **包膜与边界**：边界是否清楚，轮廓是否规则。
4. **血流模式**：是否以门部血流为主，而不是外围杂乱血流。
5. **异常征象**：是否出现圆形化、皮质增厚、门部消失、坏死、钙化等。

## 判断流程

### 1. 先看整体形态
- **更像淋巴结**：
  - 轮廓偏椭圆、梭形、豆形
  - 长轴明显大于短轴
  - 边界相对清晰
- **不太像淋巴结**：
  - 形态不规则、片状、管状、团块状
  - 与周围组织边界混杂

### 2. 再看内部回声
- **支持淋巴结**：
  - 中央可见高回声脂肪门
  - 皮质与髓质结构较清楚
  - 回声整体较均匀
- **提示异常/可疑**：
  - 脂肪门变小、受压、移位或消失
  - 皮质明显增厚，尤其是偏心性增厚
  - 回声不均、低回声增多

### 3. 看血流分布（若有彩超/能量多普勒）
- **良性/反应性更常见**：
  - 门部血流为主
  - 血流较少且走行规律
- **恶性可疑更常见**：
  - 周边血流、混合血流
  - 血流紊乱、穿支样血流

### 4. 看危险征象
若出现以下任一项，应提高“包含淋巴结且可能异常”的置信度：
- 圆形化（长短轴比下降）
- 皮质局灶或弥漫性增厚
- 脂肪门消失
- 边界毛刺或不规则
- 内部坏死/液化
- 钙化灶
- 多发融合、包膜外侵犯迹象

### 5. 结合上下文位置
- 颈部、锁骨上、腋窝等区域更常见淋巴结结构
- 若图像显示典型解剖邻近关系（血管、肌肉、脂肪间隙），可辅助定位
- 纯腺体、甲状腺结节、血管横切面、神经束等结构要与淋巴结区分

## 一个可执行的判定规则
可以把判断分成三档：

### A. 明确包含淋巴结区域
满足以下大部分特征：
- 椭圆/豆形
- 可见脂肪门
- 内部回声较均匀
- 边界清楚
- 门部血流为主

### B. 疑似包含淋巴结区域
部分特征符合，但不完整：
- 形态像淋巴结，但脂肪门不典型
- 皮质略增厚
- 回声轻度不均
- 需要更多切面或后续标注确认

### C. 不包含淋巴结区域
以下特征更明显：
- 不呈淋巴结典型形态
- 无脂肪门
- 更符合其他组织结构
- 没有可辨识的淋巴结轮廓

## 建议的标注/筛选策略
如果用于数据筛选或模型预处理，可采用下面的逻辑：

1. **先做粗筛**：根据是否存在“椭圆低回声结构 + 中央高回声门”初判。
2. **再做细筛**：检查皮质厚度、边界规则性、血流分布。
3. **输出标签**：
   - `lymph_node_present = yes/no`
   - `lymph_node_confidence = high/medium/low`
   - `lymph_node_abnormal = yes/no`
4. **低置信度样本**建议人工复核。

## 与超声图像分割任务的关系
如果后续目标是做分割，建议先做“是否包含淋巴结”的图像级过滤：
- 降低无关背景样本进入训练集
- 减少模型学习非目标结构
- 提高正样本纯度

## 参考依据
- 淋巴结超声常见良性特征：椭圆形、脂肪门保留、皮质薄且均匀、门部血流。
- 淋巴结超声常见可疑特征：圆形化、皮质增厚、脂肪门消失、外围/混合血流、坏死或钙化。

## 参考资料
- [Lymph Node Assessment with Multiparametric Ultrasound: Normal Values, Morphologic Patterns, and Diagnostic Algorithms](https://www.mdpi.com/2072-6694/18/6/1045)
- [Lymph node ultrasound in lymphoproliferative disorders: clinical characteristics and applications](https://pubmed.ncbi.nlm.nih.gov/40459902/)
- [Ultrasound of superficial lymph nodes](https://pubmed.ncbi.nlm.nih.gov/16480846/)
- [New ultrasound techniques for lymph node evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC3740414/)
- [A practical approach to imaging the axilla](https://pmc.ncbi.nlm.nih.gov/articles/PMC4376818/)
- [Ultrasound imaging of the axilla](https://pubmed.ncbi.nlm.nih.gov/37166516/)
- [LN-RADS—Retrospective Evaluation for Ultrasound Classification of Superficial Lymph Nodes](https://www.mdpi.com/2072-6694/17/12/2030)
