# 🚀 快速开始指南 - 从60分到90分+

## 📋 实现步骤速览

### 第1步：环境检查和导入修复
```bash
# 确保可以导入增强模块
python -c "from src.model.feature_enhanced import EnhancedFeatureExtraction; print('✓ OK')"
python -c "from src.model.vm_enhanced import EnhancedVelocityModule; print('✓ OK')"
```

### 第2步：阶段一 - 基础增强（目标75分）⏱️ 1-2周

**做什么**：
- ✅ 法向量估计 + 曲率加权
- ✅ 多尺度特征提取
- ✅ 更深的编码器（512维特征）
- ✅ 增强数据增强

**训练命令**：
```bash
python run.py --task configs/task/train_vm_enhanced_stage1.yaml --seed 42
```

**预期性能**：
- CD得分：70-73分
- P2S得分：72-75分
- 总分：71-74分

**调试技巧**：
```bash
# 如果显存不足，降低batch_size
sed -i 's/batch_size: 8/batch_size: 4/g' configs/task/train_vm_enhanced_stage1.yaml

# 如果收敛慢，增加学习率
# 编辑 configs/task/train_vm_enhanced_stage1.yaml
# learning_rate: 0.002  # 从0.001调到0.002
```

---

### 第3步：阶段二 - 架构升级（目标85分）⏱️ 2-3周

**做什么**：
- ✅ 启用Transformer注意力
- ✅ 自适应噪声估计
- ✅ 混合损失函数
- ✅ 迭代细化推理

**训练命令**：
```bash
python run.py --task configs/task/train_vm_enhanced_stage2.yaml --seed 42
```

**关键改动**：
```yaml
# 在 configs/model/vm_enhanced_stage2.yaml 中已配置
use_transformer: true        # Transformer注意力
use_adaptive_noise: true     # 自适应噪声
loss_type: "hybrid"          # 混合损失
use_iterative_refinement: true
```

**预期性能**：
- CD得分：78-82分
- P2S得分：80-85分
- 总分：79-83分

**性能瓶颈排查**：
```python
# 如果Transformer导致显存溢出，编辑 src/model/feature_enhanced.py
# 在 TransformerAttention.__init__ 中改为：
# num_heads: 4  # 从8降到4
# ff_dim: embedding_dim  # 从2倍降到1倍
```

---

### 第4步：阶段三 - 前沿融合（目标90+分）⏱️ 2-3周

**做什么**：
- ✅ 基于扩散的降噪
- ✅ P2S约束损失（可选高级）
- ✅ 专家混合（可选）
- ✅ 集成学习

**训练命令**：
```bash
python run.py --task configs/task/train_vm_enhanced_stage3.yaml --seed 42
```

**预期性能**：
- CD得分：86-90分
- P2S得分：88-92分
- 总分：87-91分

---

## 🔧 常用命令

### 训练
```bash
# 单个GPU训练阶段一
python run.py --task configs/task/train_vm_enhanced_stage1.yaml

# 多GPU训练（如果支持）
CUDA_VISIBLE_DEVICES=0,1 python run.py --task configs/task/train_vm_enhanced_stage1.yaml

# 从检查点继续训练
python run.py --task configs/task/train_vm_enhanced_stage1.yaml \
  --load_ckpt experiments/enhanced_stage1/checkpoint_50.pt
```

### 推理
```bash
# 推理（修改配置中的load_ckpt指向最好的权重）
python run.py --task configs/task/predict_vm_enhanced.yaml

# 使用其他模型权重推理
python run.py --task configs/task/predict_vm_enhanced.yaml \
  --load_ckpt experiments/enhanced_stage2/best.pt
```

### 评测
```bash
# 本地评测（需要安装point-cloud-utils）
pip install point-cloud-utils

python evaluate.py \
    --pred_dir ./results \
    --gt_dir ./test_gt \
    --noisy_dir ./dataset_test_noisy \
    --mesh_dir ./dataset_clean \
    --workers 8
```

---

## 💡 性能优化技巧

### 1. 动态学习率调度
```python
# 在run.py中修改学习率调度
scheduler = CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10,  # 初始周期
    T_mult=2  # 每个周期乘以2
)
```

### 2. 梯度累积（显存不足时）
```python
# 伪代码
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 混合精度训练（加速）
```python
# 如果Jittor支持，启用fp16
# 编辑 src/model/vm_enhanced.py
# jt.flags.use_fp16 = 1
```

---

## 🐛 常见问题

### Q1: "Shape mismatch"错误
```
错误：feat shape (B, N, F) 不匹配
原因：batch内点数不一致

解决：
1. 检查数据加载是否填充到相同大小
2. 确保点云采样到固定大小（50000）
```

### Q2: 显存溢出
```
解决方案按优先级：
1. 降低batch_size: 8 → 4 → 2 → 1
2. 减少特征维度: 512 → 256
3. 禁用Transformer: use_transformer: false
4. 减少KNN邻居: 32 → 16
```

### Q3: 模型不收敛
```
检查清单：
1. 学习率太高？  试试 0.0001
2. 数据增强过度？ 关闭部分增强
3. 权重初始化？    确保是标准初始化
4. Loss爆炸？      检查是否有NaN
```

### Q4: 推理太慢
```
加速方案：
1. 减少推理步数: num_inference_steps: 4 → 2
2. 使用patch分割: patch_size: 2500
3. 启用推理时间优化: use_inference_optimize: true
4. 模型量化（高级）
```

---

## 📊 性能对标表

| 模型 | CD得分 | P2S得分 | 总分 | 训练时间 |
|-----|--------|--------|------|---------|
| Baseline (60分) | 55-60 | 55-60 | 60 | - |
| 阶段一 (75分) | 70-73 | 72-75 | 72.5 | 1周 |
| 阶段二 (85分) | 78-82 | 80-85 | 81 | 2周 |
| 阶段三 (90+分) | 86-90 | 88-92 | 89 | 3周 |

---

## 🎯 建议的实施时间表

**第1-2周**：完成阶段一，达成75分
- Day 1-2：环境设置 + 理解代码
- Day 3-5：训练阶段一模型
- Day 6-10：调试 + 优化超参
- Day 11-14：验证75分达成

**第3-5周**：完成阶段二，冲击85分
- Week 1：集成Transformer + 自适应噪声
- Week 1.5：训练 + 调试
- Week 2：性能对标 + 微调

**第6-9周**：阶段三探索，争取90+分
- Week 1：研究扩散模型实现
- Week 1.5：集成 + 训练
- Week 2：对标 + 最终优化

---

## 📚 参考代码位置

| 功能 | 文件 |
|-----|------|
| 法向估计 | `src/model/feature_enhanced.py:NormalEstimator` |
| 曲率计算 | `src/model/feature_enhanced.py:CurvatureEstimator` |
| 多尺度特征 | `src/model/feature_enhanced.py:MultiScaleFeatureExtractor` |
| Transformer | `src/model/feature_enhanced.py:TransformerAttention` |
| 增强模型 | `src/model/vm_enhanced.py:EnhancedVelocityModule` |
| 扩散模型 | `src/model/vm_enhanced.py:DiffusionDenoiser` |

---

## ✅ 提交前检查清单

```bash
# 1. 验证推理输出格式
python -c "
import numpy as np
# 读取生成的denoised.npy检查
arr = np.load('results/dataset_test_noisy/shapenet/02691156/xxxxx/denoised.npy')
print(f'Shape: {arr.shape}')
print(f'Type: {arr.dtype}')
print(f'Range: [{arr.min():.4f}, {arr.max():.4f}]')
assert arr.dtype == np.float32, 'Must be float32'
assert len(arr.shape) == 2 and arr.shape[1] == 3, 'Must be (N, 3)'
"

# 2. 验证提交包结构
unzip -l result.zip | head -20

# 3. 本地评测
python evaluate.py --pred_dir ./results --gt_dir ./test_gt --noisy_dir ./dataset_test_noisy

# 4. 打包提交
cd results/dataset_test_noisy && zip -r ../../result.zip shapenet/ && cd ../../
```

---

**开始吧！一步步提升分数到90+！** 🚀
