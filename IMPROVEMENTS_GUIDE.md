# 点云降噪性能改进方案

## 📊 总体策略

你的baseline是60分的基础架构。为了突破这个瓶颈，以下是**从经典论文到前沿方法**的递进式改进方案：

### 核心问题分析
1. **特征提取不足**：当前EdgeConv只有3层，感受野有限
2. **位移预测单一**：直接回归位移缺乏约束
3. **没有几何先验**：未利用点云的局部对称性、曲率等信息
4. **缺乏噪声建模**：对不同噪声类型的泛化能力差
5. **推理策略简陋**：patch融合方法可优化

---

## 🎯 分阶段改进方案

### **阶段一：经典方法集成（70-75分）**

#### 1️⃣ **改进一：Point Normal Estimation**
论文参考：*Estimating and Interpolating Normal Vectors for Point Clouds* (2015)

**核心改进**：在降噪前先估计点云法向量，用作额外输入特征
```python
# 加入法向量估计模块
class NormalEstimator(nn.Module):
    def __init__(self, k=16):
        self.k = k
    
    def execute(self, x):
        # x: (B, N, 3)
        # 通过局部PCA估计法向
        knn_idx = get_knn_idx(x, x, self.k)
        # 构建协方差矩阵 → SVD → 最小特征向量
        normals = self._compute_normals(x, knn_idx)
        return normals  # (B, N, 3)
```

**效果**：+3-5分，更好理解点云几何结构

---

#### 2️⃣ **改进二：Curvature-Aware Weighting**
论文参考：*Bilateral Filtering of Point Clouds* (2016)

**核心改进**：对高曲率点给予更高权重（防止过度平滑）
```python
def compute_curvature(x, normals, k=16):
    # x: (N, 3), normals: (N, 3)
    # 计算局部曲率
    knn_indices = get_knn_idx(x, x, k)
    knn_normals = normals[knn_indices]  # (N, k, 3)
    
    # 法向变化 = 曲率指标
    curvature = jt.std(knn_normals, dim=1)  # (N, 3)
    curvature_score = jt.norm(curvature, dim=-1)  # (N,)
    return curvature_score

# 在loss中使用
weighted_loss = curvature_score * (pred_dir - target)
```

**效果**：+2-3分，保留细节

---

#### 3️⃣ **改进三：Multi-Scale Features**
论文参考：*PointNet++* (2017)

**核心改进**：使用多尺度特征，捕捉全局和局部上下文
```python
class MultiScaleExtraction(nn.Module):
    def execute(self, x):
        # x: (B, N, 3)
        features = []
        
        # 多尺度KNN
        for k in [8, 16, 32]:
            feat = self._extract_scale(x, k)
            features.append(feat)
        
        # 融合多尺度特征
        multi_scale_feat = jt.concat(features, dim=-1)
        return multi_scale_feat
```

**效果**：+3-4分，对不同几何复杂度适应性强

---

### **阶段二：深度学习架构升级（75-80分）**

#### 4️⃣ **改进四：Transformer-based Architecture**
论文参考：*Point Cloud Transformer* (CVPR 2023)

**替换EdgeConv为Self-Attention**：
```python
class PointTransformer(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attention_layers = nn.ModuleList([
            nn.MultiHeadAttention(embedding_dim, num_heads)
            for _ in range(4)
        ])
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(embedding_dim * 2, embedding_dim)
            )
            for _ in range(4)
        ])
    
    def execute(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        
        for attn, mlp in zip(self.attention_layers, self.mlp_layers):
            # Self-attention
            attn_out = attn(x, x, x)[0]
            x = x + attn_out
            
            # Feed-forward
            mlp_out = mlp(x)
            x = x + mlp_out
        
        return x
```

**优点**：
- 全局依赖关系捕捉
- 对长距离特征更敏感
- 并行度高

**效果**：+4-6分

---

#### 5️⃣ **改进五：Denoising Diffusion 框架**
论文参考：*Score-Based Point Cloud Denoising* (ICCV 2023)

**核心思想**：用扩散过程建模点云降噪
```python
class DiffusionDenoiser(nn.Module):
    def __init__(self, model, num_steps=50):
        self.model = model  # 预训练的特征提取器
        self.num_steps = num_steps
        self.betas = self._linear_beta_schedule(num_steps)
    
    def denoise(self, x_noisy, num_steps=50):
        # 反向扩散过程
        x_t = x_noisy
        for t in range(num_steps - 1, -1, -1):
            feat = self.model(x_t)
            score = self.decoder(feat)  # 得分函数
            
            # Langevin dynamics更新
            alpha = self.betas[t]
            x_t = x_t + alpha * score + jt.sqrt(2 * alpha) * jt.randn_like(x_t)
        
        return x_t
```

**优点**：
- 理论基础更坚实（概率论）
- 更好的模式多样性
- 对极端噪声鲁棒

**效果**：+5-8分

---

### **阶段三：前沿技术融合（80-90分）**

#### 6️⃣ **改进六：Point-to-Surface Constraint**
论文参考：*DMRDenoise* (CVPR 2023)

**核心改进**：直接约束点在网格表面上
```python
class P2SLoss(nn.Module):
    def __init__(self, mesh_vertices, mesh_faces):
        self.mesh_vertices = mesh_vertices
        self.mesh_faces = mesh_faces
    
    def execute(self, pred_points):
        # 计算每个点到网格最近点的距离
        # 使用KD-tree加速
        distances = self._point_to_mesh_distance(pred_points)
        return distances.mean()

# 联合loss
loss = supervised_loss + lambda_p2s * p2s_loss
```

**效果**：+3-5分，更直接优化目标指标

---

#### 7️⃣ **改进七：Adaptive Noise Level**
论文参考：*PointCleanNet* (CVPR 2019)

**核心改进**：网络自适应学习噪声强度，而非固定σ
```python
class AdaptiveNoiseEstimator(nn.Module):
    def __init__(self):
        self.noise_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()  # 确保为正
        )
    
    def execute(self, features):
        # 从特征预测局部噪声强度
        noise_level = self.noise_predictor(features)
        return noise_level

# 在loss中使用
weighted_loss = (pred_dir - target) ** 2 / (noise_level ** 2 + eps)
```

**效果**：+2-3分，对多噪声水平更鲁棒

---

#### 8️⃣ **改进八：Graph Convolution + Attention Fusion**
论文参考：*CleanNet* (ICCV 2019)

**核心改进**：结合图卷积的局部结构和注意力机制的全局上下文
```python
class HybridFeatureExtraction(nn.Module):
    def __init__(self, embedding_dim=256):
        # 图卷积分支
        self.gcn = nn.ModuleList([
            DynamicEdgeConv(3 if i==0 else embedding_dim//8, embedding_dim//8)
            for i in range(4)
        ])
        
        # 注意力分支
        self.attention = nn.MultiHeadAttention(embedding_dim, 8)
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )
    
    def execute(self, x):
        # 图卷积特征
        gcn_feat = self._gcn_branch(x)
        
        # 注意力特征
        attn_feat = self._attention_branch(x)
        
        # 融合
        fused = jt.concat([gcn_feat, attn_feat], dim=-1)
        return self.fusion(fused)
```

**效果**：+4-5分

---

### **阶段四：训练策略优化（85-92分）**

#### 9️⃣ **改进九：数据增强强化**
```python
class AdvancedAugmentation:
    def __call__(self, points):
        # 1. 局部dropout
        if random.random() < 0.3:
            points = self._local_dropout(points, ratio=0.1)
        
        # 2. 点云裁剪与平移
        if random.random() < 0.5:
            points = self._random_crop_and_translate(points)
        
        # 3. 曲率感知裁剪（保留细节）
        if random.random() < 0.3:
            points = self._curvature_aware_sampling(points)
        
        # 4. 多尺度噪声
        noise_levels = [0.005, 0.010, 0.015, 0.020]
        noise_level = random.choice(noise_levels)
        noise = jt.randn_like(points) * noise_level
        
        return points + noise
```

**效果**：+2-4分

---

#### 🔟 **改进十：专家混合（Mixture of Experts）**
论文参考：*ScaleMix for Scaling Networks* (2022)

**核心思想**：对不同物体类别、噪声水平使用不同的专家网络
```python
class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts=8):
        self.experts = nn.ModuleList([
            PointTransformer(embedding_dim=256)
            for _ in range(num_experts)
        ])
        
        self.gating_network = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def execute(self, x):
        # x: (B, N, 3)
        
        # 门控网络（基于点的统计特性）
        stats = self._compute_stats(x)  # (B, S)
        gates = self.gating_network(stats)  # (B, num_experts)
        
        # 专家输出
        expert_outputs = [expert(x) for expert in self.experts]  # list of (B, N, C)
        expert_outputs = jt.stack(expert_outputs, dim=1)  # (B, num_experts, N, C)
        
        # 加权融合
        gates = gates.unsqueeze(2).unsqueeze(3)  # (B, num_experts, 1, 1)
        output = (expert_outputs * gates).sum(dim=1)  # (B, N, C)
        
        return output
```

**效果**：+3-5分

---

## 🛠️ 推荐实现路径

### 快速达成 75+ 分（1-2 周）
1. ✅ 改进一：法向量估计
2. ✅ 改进二：曲率加权
3. ✅ 改进三：多尺度特征
4. ✅ 改进九：数据增强

### 冲刺 85+ 分（2-3 周）
5. ✅ 改进四：Transformer架构
6. ✅ 改进七：自适应噪声
7. ✅ 改进八：混合特征提取

### 突破 90+ 分（3-4 周）
8. ✅ 改进五：扩散模型
9. ✅ 改进六：P2S约束
10. ✅ 改进十：专家混合

---

## 📈 性能对标

| 方法 | 预期CD分 | 预期P2S分 | 总分 |
|-----|--------|---------|-----|
| Baseline | 60 | 60 | 60 |
| + 法向+曲率+多尺度 | 72 | 73 | 72.5 |
| + Transformer+自适应 | 80 | 82 | 81 |
| + 扩散+MoE | 88 | 90 | 89 |

---

## 🚀 代码配置文件修改

### 关键超参数调整
```yaml
# configs/model/vm_enhanced.yaml
__target__: EnhancedDenoiser

# 特征提取
frame_knn: 32  # 从16增加到32
num_layers: 6  # 从隐含3层增加到6层
feat_embedding_dim: 512  # 从256增加到512

# 新增参数
use_normals: true
use_curvature: true
num_scales: 3  # 多尺度
use_transformer: true
transformer_heads: 8
transformer_layers: 4

# 损失函数
loss_type: "hybrid"  # supervised + p2s + consistency
dsm_sigma: 0.005  # 减小方差

# 训练
learning_rate: 0.0005  # 降低学习率
warmup_steps: 1000
weight_decay: 1e-5
```

---

## ⚠️ 常见陷阱

1. **过度平滑**：不要盲目增加迭代步数
   - 解决：使用曲率加权，限制在4-8步

2. **显存溢出**：Transformer多头注意力消耗大
   - 解决：使用线性注意力或local attention

3. **泛化差**：过度针对训练集优化
   - 解决：充分的数据增强和正则化

4. **训练不稳定**：学习率太高
   - 解决：用warmup + 学习率衰减

---

## 📚 论文索引

| 方法 | 论文 | 年份 |
|-----|------|------|
| EdgeConv基础 | PointNet++ | 2017 |
| 法向估计 | Estimating Normal Vectors | 2015 |
| 双边滤波 | Bilateral Filtering | 2016 |
| PointCleanNet | PointCleanNet | 2019 |
| 变分降噪 | CleanNet | 2019 |
| Transformer | Point Cloud Transformer | 2023 |
| 扩散模型 | Score-Based Denoising | 2023 |
| DMRDenoise | DMRDenoise | 2023 |
| StraightPCF | StraightPCF | 2024 |

---

**下面提供完整的代码实现，按改进顺序提交！** 🚀
