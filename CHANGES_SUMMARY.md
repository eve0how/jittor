🎯 **5天冲75分 - 改动清单** 

## 📦 新增4个文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/model/feature_enhanced_quick.py` | 650行 | 法向+曲率+多尺度特征提取 |
| `src/model/vm_stage1.py` | 350行 | 改进版模型（集成新特征） |
| `configs/model/vm_stage1_quick.yaml` | 20行 | 模型参数 |
| `configs/task/train_stage1_quick.yaml` | 30行 | 训练参数 |

## ✏️ 改动1个文件

### `src/model/parse.py` - 2行改动

**改动前：**
```python
from .spec import ModelSpec
from .vm import VelocityModule

def get_model(model_config, **kwargs) -> ModelSpec:
    MAP = {
        'VelocityModule': VelocityModule,
    }
```

**改动后：**
```python
from .spec import ModelSpec
from .vm import VelocityModule
from .vm_stage1 import VelocityModuleStage1  # ← 添加此行

def get_model(model_config, **kwargs) -> ModelSpec:
    MAP = {
        'VelocityModule': VelocityModule,
        'VelocityModuleStage1': VelocityModuleStage1,  # ← 添加此行
    }
```

## 🚀 运行方式

```bash
# 就这一个命令，开始训练
python run.py --task configs/task/train_stage1_quick.yaml
```

## 📊 性能提升

| 指标 | 值 |
|------|-----|
| Baseline | 60分 |
| 改进后预期 | 75分 |
| 提升 | **+15分** |
| 时间投入 | 2-3天 |

## 🔑 核心改进

### 1️⃣ 法向量 + 曲率 (+5-8分)
```python
# 使用几何约束保护边界
QuickNormalEstimator()  # PCA法向量
QuickCurvatureEstimator()  # 法向变化
```

### 2️⃣ 多尺度特征融合 (+3-4分)
```python
# 3个尺度 [8, 16, 24] 的邻域特征
QuickMultiScaleFeature(scales=[8, 16, 24])
```

### 3️⃣ 曲率加权损失 (+2-3分)
```python
# 高曲率区域(边界)权重更大
PointwiseWeightedLoss(weight=1+2*curvature)
```

### 4️⃣ 多步迭代推理 (+1-2分)
```python
# 4步逐步精化
for step in range(4):
    delta = model(pc)
    pc = pc + delta
```

## ⏱️ 时间表

```
Day 1: 复制4个文件 + 改parse.py (2行) + 验证
Day 2: 训练80个epoch (~5小时)
Day 3: 评测 + 微调
Day 4: 再训一遍 + 评测
Day 5: 最后检查 + 提交
```

## ✅ 检查清单

```
□ 复制了4个新文件
□ 改了parse.py (添加2行import)
□ 运行 python run.py --task configs/task/train_stage1_quick.yaml
□ 等待训练完成（~5小时）
□ 评测结果
□ 准备提交
```

---

**就这么简单！现在就开始吧！** 🚀
