"""
快速高效的特征提取模块 - 为75分优化 (简化版，已修复 API 兼容性)
核心改进：法向量 + 曲率 + 多尺度 + KNN融合
设计目标：最大化性能/时间比，避免 Jittor API 陷阱
"""
import jittor as jt
import jittor.nn as nn
import math


def pairwise_distance(x):
    """
    计算成对平方距离
    x: (B, N, C) -> (B, N, N)
    使用: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i·x_j
    """
    # (B, N, C) -> (B, N, 1) 每个点自身的模长平方
    sq = (x ** 2).sum(dim=2, keepdim=True)  # (B, N, 1)
    
    # 计算所有对的内积: x @ x^T -> (B, N, N)
    dot_product = jt.matmul(x, x.transpose(0, 2, 1))  # (B, N, N)
    
    # 距离矩阵: sq_i + sq_j - 2*dot(x_i, x_j)
    # sq.transpose(0, 2, 1) -> (B, 1, N)
    dist = sq + sq.transpose(0, 2, 1) - 2 * dot_product  # (B, N, N)
    
    # 确保非负（数值稳定性）
    dist = jt.clip(dist, min_v=0.0)
    return dist


def knn_indices(dist, k):
    """
    从距离矩阵获取 top-k 最近邻索引
    dist: (B, N, N) - 距离矩阵
    k: 邻域大小
    返回: (B, N, k) - 索引矩阵
    最简单稳定的实现：直接用基本操作
    """
    B, N, _ = dist.shape
    
    # 最保险的方法：逐批次逐点处理
    # 虽然不是最优效率，但最稳定
    all_indices = []
    
    for b in range(B):
        batch_indices = []
        for n in range(N):
            dist_row = dist[b, n]  # (N,)
            
            # 创建候选列表：(距离值, 索引)
            # 用数值方式找最小 k 个
            knn_idx_list = []
            remaining_idx = list(range(N))
            remaining_dist = [dist_row[i] for i in range(N)]
            
            for _ in range(min(k, N)):
                # 找最小距离的位置
                min_pos = 0
                min_val = remaining_dist[0]
                for pos in range(1, len(remaining_dist)):
                    # 注意：这里比较可能涉及 Jittor 标量
                    # 用一个技巧：sum() 后比较
                    if remaining_dist[pos] < min_val:
                        min_val = remaining_dist[pos]
                        min_pos = pos
                
                # 记录对应的原始索引
                knn_idx_list.append(remaining_idx[min_pos])
                
                # 删除该元素
                remaining_idx.pop(min_pos)
                remaining_dist.pop(min_pos)
            
            # 转为 Jittor 数组
            knn_arr = jt.array(knn_idx_list).int32()
            batch_indices.append(knn_arr)
        
        # 堆叠为 (N, k)
        batch_result = jt.stack(batch_indices, dim=0)  # (N, k)
        all_indices.append(batch_result)
    
    # 堆叠为 (B, N, k)
    result = jt.stack(all_indices, dim=0)
    return result


class QuickNormalEstimator(nn.Module):
    """快速法向量估计 - 简化版"""
    
    def __init__(self, k=8):
        super().__init__()
        self.k = k
    
    def execute(self, xyz):
        """xyz: (B, N, 3) -> (B, N, 3) 单位法向量"""
        B, N, _ = xyz.shape
        
        dist = pairwise_distance(xyz)  # (B, N, N)
        indices = knn_indices(dist, self.k)  # (B, N, k)
        
        batch_idx = jt.arange(B).reshape(B, 1, 1).expand(B, N, self.k)
        neighbors = xyz[batch_idx, indices]  # (B, N, k, 3)
        center = xyz.unsqueeze(2)  # (B, N, 1, 3)
        rel = neighbors - center  # (B, N, k, 3)
        
        # 简化：直接用相对坐标的平均作为法向
        normal = rel.mean(dim=2)  # (B, N, 3)
        normal = normal / (jt.norm(normal, dim=-1, keepdim=True) + 1e-8)
        return normal


class QuickCurvatureEstimator(nn.Module):
    """快速曲率估计 - 简化版"""
    
    def __init__(self, k=8):
        super().__init__()
        self.k = k
    
    def execute(self, xyz):
        """xyz: (B, N, 3) -> (B, N, 1) 曲率"""
        B, N, _ = xyz.shape
        
        dist = pairwise_distance(xyz)  # (B, N, N)
        indices = knn_indices(dist, self.k)  # (B, N, k)
        
        batch_idx = jt.arange(B).reshape(B, 1, 1).expand(B, N, self.k)
        neighbors = xyz[batch_idx, indices]  # (B, N, k, 3)
        center = xyz.unsqueeze(2)  # (B, N, 1, 3)
        rel = neighbors - center  # (B, N, k, 3)
        
        # 曲率 = 邻域平均距离
        dists_to_neighbors = jt.norm(rel, dim=-1)  # (B, N, k)
        curvature = dists_to_neighbors.mean(dim=-1, keepdim=True)  # (B, N, 1)
        # 归一化
        eps = curvature.mean() + 1e-8
        curvature = jt.clip(curvature / eps, min_v=0.0, max_v=1.0)
        return curvature


class SimpleMultiScaleFeature(nn.Module):
    """简化多尺度特征 - 邻域统计"""
    
    def __init__(self, out_channels=252):
        super().__init__()
        self.out_channels = out_channels
        self.fc = nn.Sequential(
            nn.Linear(27, 128),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Linear(128, out_channels)
        )
    
    def execute(self, xyz):
        """xyz: (B, N, 3) -> (B, N, out_channels)"""
        B, N, C = xyz.shape
        dist = pairwise_distance(xyz)  # (B, N, N)
        
        features = []
        for k in [4, 8, 16]:
            indices = knn_indices(dist, k)  # (B, N, k)
            batch_idx = jt.arange(B).reshape(B, 1, 1).expand(B, N, k)
            neighbors = xyz[batch_idx, indices]  # (B, N, k, 3)
            
            # 统计：最大、最小、平均
            feat_max = neighbors.max(dim=2)[0]  # (B, N, 3)
            feat_min = neighbors.min(dim=2)[0]  # (B, N, 3)
            feat_mean = neighbors.mean(dim=2)   # (B, N, 3)
            
            feat = jt.concat([feat_max, feat_mean, feat_min], dim=-1)  # (B, N, 9)
            features.append(feat)
        
        feat = jt.concat(features, dim=-1)  # (B, N, 27)
        feat = feat.reshape(-1, 27)
        feat = self.fc(feat)
        feat = feat.reshape(B, N, -1)
        return feat


class QuickFeatureExtraction(nn.Module):
    """快速综合特征提取"""
    
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        self.normal_est = QuickNormalEstimator(k=8)
        self.curvature_est = QuickCurvatureEstimator(k=8)
        self.multi_scale = SimpleMultiScaleFeature(out_channels=252)
    
    def execute(self, xyz):
        """xyz: (B, N, 3) -> (B, N, 256)"""
        normals = self.normal_est(xyz)  # (B, N, 3)
        curvature = self.curvature_est(xyz)  # (B, N, 1)
        multi_feat = self.multi_scale(xyz)  # (B, N, 252)
        features = jt.concat([normals, curvature, multi_feat], dim=-1)
        return features


class PointwiseWeightedLoss(nn.Module):
    """逐点加权损失"""
    
    def __init__(self):
        super().__init__()
    
    def execute(self, pred, target, curvature=None):
        """pred/target: (B, N, 3), curvature: (B, N, 1) or None"""
        loss = jt.sum((pred - target) ** 2, dim=-1, keepdim=True)  # (B, N, 1)
        if curvature is not None:
            weight = 1.0 + 2.0 * curvature
            loss = loss * weight
        return loss.mean()


__all__ = [
    'QuickNormalEstimator',
    'QuickCurvatureEstimator', 
    'SimpleMultiScaleFeature',
    'QuickFeatureExtraction',
    'PointwiseWeightedLoss'
]
