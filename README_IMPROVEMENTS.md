# 🎯 点云降噪赛题 - 完整改进方案

## 📚 文档导航

### 🚀 快速开始（推荐首先阅读）
1. **[QUICK_START.md](./QUICK_START.md)** - 快速开始指南
   - 训练命令一行运行
   - 常见问题Q&A
   - 性能对标表

2. **[COMPLETE_PLAN.md](./COMPLETE_PLAN.md)** - 完整实施方案
   - 三阶段详细规划（75分→85分→90+分）
   - 每周任务分解
   - 时间表安排

### 📖 深度学习
3. **[IMPROVEMENTS_GUIDE.md](./IMPROVEMENTS_GUIDE.md)** - 改进方案详解
   - 10种改进方法详细说明
   - 每个方法的代码样例
   - 论文参考索引
   - 性能收益分析

4. **[INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)** - 代码集成说明
   - 如何集成新模块
   - API对应关系
   - 各个文件的作用

---

## 📦 核心文件清单

### 代码模块
```
src/model/
├── feature_enhanced.py      ⭐ 增强特征提取模块
│   ├── NormalEstimator           - 法向量估计
│   ├── CurvatureEstimator        - 曲率计算
│   ├── MultiScaleFeatureExtractor - 多尺度特征
│   ├── TransformerAttention      - Transformer注意力
│   ├── GraphConvolutionLayer     - 图卷积层
│   └── EnhancedFeatureExtraction - 综合特征提取
│
└── vm_enhanced.py           ⭐ 增强模型
    ├── AdaptiveNoiseEstimator   - 自适应噪声估计
    ├── EnhancedVelocityModule   - 增强位移模块
    └── DiffusionDenoiser        - 扩散降噪模型
```

### 配置文件
```
configs/
├── model/
│   ├── vm_enhanced_stage1.yaml   - 阶段一配置 (75分)
│   ├── vm_enhanced_stage2.yaml   - 阶段二配置 (85分)
│   └── vm_enhanced_stage3.yaml   - 阶段三配置 (90+分)
│
└── task/
    ├── train_vm_enhanced_stage1.yaml   - 阶段一训练
    ├── train_vm_enhanced_stage2.yaml   - 阶段二训练
    └── predict_vm_enhanced.yaml        - 推理配置
```

---

## 🎯 三阶段方案概览

### 🥇 阶段一：基础增强 (75分)
**时间**: 1-2周 | **难度**: ⭐☆☆☆☆

**核心改进**:
- ✅ 法向量估计 (+3-5分)
- ✅ 曲率加权 (+2-3分)
- ✅ 多尺度特征 (+3-4分)
- ✅ 数据增强强化 (+2-3分)

**训练**:
```bash
python run.py --task configs/task/train_vm_enhanced_stage1.yaml
```

---

### 🥈 阶段二：架构升级 (85分)
**时间**: 2-3周 | **难度**: ⭐⭐⭐☆☆

**核心改进**:
- ✅ Transformer注意力 (+4-6分)
- ✅ 自适应噪声估计 (+2-3分)
- ✅ 混合损失函数 (+2-3分)
- ✅ 迭代细化推理 (+1-2分)

**训练**:
```bash
python run.py --task configs/task/train_vm_enhanced_stage2.yaml
```

---

### 🥉 阶段三：前沿融合 (90+分)
**时间**: 2-3周 | **难度**: ⭐⭐⭐⭐⭐

**核心改进**:
- ✅ 扩散模型框架 (+5-8分)
- ✅ P2S约束损失 (+3-5分)
- ✅ 专家混合 (+2-4分)

**训练**:
```bash
python run.py --task configs/task/train_vm_enhanced_stage3.yaml
```

---

## 🚀 立即开始

### 第一步：诊断环境
```bash
python diagnose.py
```

如果所有检查通过，则可以继续。如果有失败，查阅对应的错误信息。

### 第二步：选择阶段
```bash
# 推荐从阶段一开始
python run.py --task configs/task/train_vm_enhanced_stage1.yaml

# 或者使用自定义配置
python run.py --task configs/task/train_vm_enhanced_stage1.yaml \
  --seed 42 \
  --load_ckpt <path_to_checkpoint>  # 可选：从检查点继续
```

### 第三步：监控训练
```bash
# 在另一个终端查看日志
tail -f experiments/enhanced_stage1/logs.txt
```

### 第四步：推理和提交
```bash
# 修改配置中的load_ckpt指向最好的权重
python run.py --task configs/task/predict_vm_enhanced.yaml

# 打包提交
cd results/dataset_test_noisy
zip -r ../../result.zip shapenet/
```

---

## 📈 性能目标

| 阶段 | CD得分 | P2S得分 | 总分 | 相比baseline |
|------|--------|--------|------|-------------|
| Baseline | 55-60 | 55-60 | 60 | - |
| 🥇 一 | 70-73 | 72-75 | 72.5 | +12.5 |
| 🥈 二 | 78-82 | 80-85 | 81 | +21 |
| 🥉 三 | 86-90 | 88-92 | 89 | +29 |

---

## 🔧 常用命令速查

### 训练
```bash
# 默认配置训练
python run.py --task configs/task/train_vm_enhanced_stage1.yaml

# 自定义seed
python run.py --task configs/task/train_vm_enhanced_stage1.yaml --seed 123

# 从检查点加载
sed -i 's/load_ckpt:/load_ckpt: experiments\/stage1\/checkpoint_50.pt/g' \
  configs/task/train_vm_enhanced_stage1.yaml
```

### 推理
```bash
# 使用最好的权重推理
sed -i 's/load_ckpt:/load_ckpt: experiments\/enhanced_stage2\/best.pt/g' \
  configs/task/predict_vm_enhanced.yaml
python run.py --task configs/task/predict_vm_enhanced.yaml
```

### 评测
```bash
# 本地评测（CD+P2S）
python evaluate.py \
  --pred_dir ./results \
  --gt_dir ./test_gt \
  --noisy_dir ./dataset_test_noisy \
  --mesh_dir ./dataset_clean \
  --workers 8

# 仅评测CD（无需网格）
python evaluate.py \
  --pred_dir ./results \
  --noisy_dir ./dataset_test_noisy \
  --workers 8
```

---

## 💡 关键改进解析

### 1️⃣ 法向量估计
```python
# 为什么重要？
# - 识别点云的局部几何方向
# - 检测尖锐边缘
# - 指导降噪方向

# 实现方式：
normals = NormalEstimator(k=16)(point_cloud)
# 基于PCA的局部法向量估计
```

### 2️⃣ 曲率加权
```python
# 为什么重要？
# - 高曲率点（边缘）应更仔细处理
# - 避免过度平滑
# - 保留细节特征

# 实现方式：
curvature = CurvatureEstimator(k=16)(points, normals)
loss = base_loss * (1 + curvature)  # 高曲率点权重更高
```

### 3️⃣ 多尺度特征
```python
# 为什么重要？
# - 结合局部（8邻居）和全局（32邻居）信息
# - 对不同尺度几何结构鲁棒
# - 灵感来自PointNet++

# 实现方式：
multi_scale = MultiScaleFeatureExtractor(scales=[8, 16, 32])
features = multi_scale(points)
```

### 4️⃣ Transformer注意力
```python
# 为什么重要？
# - 超越KNN邻居的局限
# - 捕捉长距离几何相关性
# - 全局依赖建模

# 实现方式：
transformer = TransformerAttention(num_heads=8, num_layers=4)
enhanced = transformer(local_features)
```

### 5️⃣ 自适应噪声
```python
# 为什么重要？
# - 不同区域噪声强度不同
# - 对异构噪声更鲁棒
# - 自动学习局部噪声强度

# 实现方式：
noise_level = noise_estimator(features)  # (B, N, 1)
weighted_loss = (pred - target)² / noise_level²
```

---

## ⚙️ 超参数调整指南

### 显存不足？
```yaml
# 方案1：降低batch_size
batch_size: 8 → 4

# 方案2：降低特征维度
feat_embedding_dim: 512 → 256

# 方案3：禁用Transformer
use_transformer: true → false

# 方案4：减少KNN邻居
frame_knn: 32 → 16
```

### 模型不收敛？
```yaml
# 降低学习率
learning_rate: 0.001 → 0.0001

# 增加warmup步数
warmup_steps: 1000 → 5000

# 增加正则项
weight_decay: 1e-5 → 1e-4
```

### 推理太慢？
```yaml
# 减少推理步数
num_inference_steps: 4 → 2

# 启用patch推理
use_patch_inference: false → true
patch_size: 2500
```

---

## 📊 论文参考

| 方法 | 论文 | 年份 | 备注 |
|-----|------|------|------|
| 多层PointNet | PointNet++ | 2017 | 多尺度特征基础 |
| 法向估计 | Estimating Normal Vectors | 2015 | 经典方法 |
| 双边滤波 | Bilateral Filtering | 2016 | 曲率感知 |
| PointCleanNet | PointCleanNet | 2019 | 点云降噪经典 |
| 变分降噪 | CleanNet | 2019 | 位移预测 |
| Transformer | Point Cloud Transformer | 2023 | 自注意力架构 |
| 扩散模型 | Score-Based Denoising | 2023 | 前沿方法 |
| StraightPCF | StraightPCF | 2024 | Baseline参考 |

---

## 🐛 调试技巧

### 查看中间特征
```python
# 在model中添加
def get_features(self, x):
    return self.encoder(x)

# 在推理中使用
features = model.get_features(point_cloud)
print(f"Feature shape: {features.shape}")
print(f"Feature stats: min={features.min()}, max={features.max()}")
```

### 可视化降噪结果
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载结果
noisy = np.load('dataset_test_noisy/xxx/noisy.npy')
denoised = np.load('results/xxx/denoised.npy')
clean = np.load('test_gt/xxx/clean.npy')

# 可视化
fig = plt.figure(figsize=(15, 5))
for i, (pc, title) in enumerate([
    (noisy, 'Noisy'), 
    (denoised, 'Denoised'), 
    (clean, 'Clean')
]):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    ax.scatter(*pc.T, s=1)
    ax.set_title(title)
plt.show()
```

### 性能分析
```bash
# 查看单个样本的指标
python -c "
import numpy as np
from src.data.evaluate import compute_cd, compute_p2s

pred = np.load('results/xxx/denoised.npy')
noisy = np.load('dataset_test_noisy/xxx/noisy.npy')
clean = np.load('test_gt/xxx/clean.npy')

cd_pred = compute_cd(pred, clean)
cd_noisy = compute_cd(noisy, clean)
cd_score = max(0, 100 * (1 - cd_pred/cd_noisy))

print(f'CD Score: {cd_score:.1f}')
"
```

---

## 📞 常见问题

### Q: 从哪个阶段开始？
**A**: 推荐从阶段一开始，逐步提升。这样可以稳定地达到75分，然后持续改进。

### Q: 需要多长时间训练？
**A**: 
- 阶段一：每个epoch ~1小时，推荐训练100个epoch（4-5天）
- 阶段二：每个epoch ~1.5小时，推荐训练150个epoch（9天）
- 阶段三：每个epoch ~2小时，推荐训练200个epoch（17天）

### Q: 可以同时开发多个阶段吗？
**A**: 可以，但建议首先完成一个阶段的训练和验证，再进行下一个。

### Q: 如何快速验证改进是否有效？
**A**: 
1. 在小数据集上训练（100个样本）
2. 评测CD/P2S分数
3. 与baseline对比
4. 确认后再进行全量训练

---

## 🎓 学习资源

### 推荐阅读顺序
1. 本README快速了解框架
2. QUICK_START.md学习如何运行
3. IMPROVEMENTS_GUIDE.md理解每个改进的原理
4. COMPLETE_PLAN.md规划时间表
5. INTEGRATION_GUIDE.md深入代码细节

### 关键概念
- **点云**：三维空间中的点集，表示物体表面
- **噪声**：点偏离真实表面的偏移
- **降噪**：恢复干净点云的过程
- **Chamfer Distance**：两个点云之间的距离度量
- **Point-to-Surface Distance**：点到网格表面的距离

---

## 🏆 最后的话

这是一个**循序渐进的改进方案**：

✅ **阶段一** (75分) - 基础但有效
- 快速看到效果
- 建立信心

✅ **阶段二** (85分) - 深度学习升级
- 更强大的架构
- 更好的泛化能力

✅ **阶段三** (90+分) - 前沿技术
- 扩散模型等新方法
- 理论和实践结合

**关键是坚持和迭代。每个阶段都能为下一个阶段奠定基础。**

---

**现在就开始你的改进之旅吧！** 🚀

有任何问题，参考[QUICK_START.md](./QUICK_START.md)的常见问题部分。

祝你成功！🎉
