"""
增强的特征提取模块
包含：法向估计、曲率计算、多尺度特征、Transformer注意力
"""
from typing import Optional, Tuple
import jittor as jt
from jittor import nn
import numpy as np


def get_knn_idx(x, y, k):
    """
    获取KNN索引
    x: (B, N, 3) or (N, 3)
    y: (B, M, 3) or (M, 3)
    返回: (B, N, k) 或 (N, k) 的索引
    """
    if x.ndim == 2:
        # (N, 3) 情况
        dist = jt.cdist(x, y)  # (N, M)
        _, knn_idx = jt.topk(dist, k, dim=1, largest=False)
        return knn_idx  # (N, k)
    else:
        # (B, N, 3) 情况
        B, N, _ = x.shape
        B, M, _ = y.shape
        x_flat = x.reshape(B * N, 3)  # (B*N, 3)
        y_flat = y.reshape(B * M, 3)  # (B*M, 3)
        
        dist = jt.cdist(x_flat, y_flat)  # (B*N, B*M)
        _, knn_idx = jt.topk(dist, k, dim=1, largest=False)
        knn_idx = knn_idx.reshape(B, N, k)
        return knn_idx


class NormalEstimator(nn.Module):
    """估计点云法向量（基于本地PCA）"""
    
    def __init__(self, k: int = 16):
        super().__init__()
        self.k = k
    
    def execute(self, x):
        """
        x: (B, N, 3) 或 (N, 3)
        返回: (B, N, 3) 或 (N, 3) 的法向量
        """
        if x.ndim == 2:
            return self._estimate_normals_2d(x)
        else:
            return self._estimate_normals_3d(x)
    
    def _estimate_normals_2d(self, x):
        # x: (N, 3)
        N = x.shape[0]
        
        # 获取KNN
        knn_idx = get_knn_idx(x, x, self.k + 1)[:, 1:]  # (N, k) 排除自己
        
        normals = []
        for i in range(N):
            neighbors = x[knn_idx[i]]  # (k, 3)
            
            # 中心化
            centroid = neighbors.mean(dim=0, keepdims=True)
            centered = neighbors - centroid
            
            # SVD
            U, S, V = jt.svd(centered)
            # 最小特征向量（最后一个）
            normal = V[:, -1]
            normals.append(normal)
        
        normals = jt.stack(normals, dim=0)  # (N, 3)
        return normals
    
    def _estimate_normals_3d(self, x):
        # x: (B, N, 3)
        B, N, _ = x.shape
        x_flat = x.reshape(B * N, 3)
        
        # 获取KNN（对于合并后的batch）
        knn_idx = get_knn_idx(x_flat, x_flat, self.k + 1)[:, 1:]  # (B*N, k)
        
        normals_flat = []
        for i in range(B * N):
            neighbors = x_flat[knn_idx[i]]  # (k, 3)
            centroid = neighbors.mean(dim=0, keepdims=True)
            centered = neighbors - centroid
            
            # SVD
            U, S, V = jt.svd(centered)
            normal = V[:, -1]
            normals_flat.append(normal)
        
        normals_flat = jt.stack(normals_flat, dim=0)  # (B*N, 3)
        normals = normals_flat.reshape(B, N, 3)
        return normals


class CurvatureEstimator(nn.Module):
    """估计点云曲率"""
    
    def __init__(self, k: int = 16):
        super().__init__()
        self.k = k
    
    def execute(self, x, normals):
        """
        x: (B, N, 3) 或 (N, 3)
        normals: (B, N, 3) 或 (N, 3)
        返回: (B, N) 或 (N,) 的曲率
        """
        if x.ndim == 2:
            return self._compute_curvature_2d(x, normals)
        else:
            return self._compute_curvature_3d(x, normals)
    
    def _compute_curvature_2d(self, x, normals):
        # x: (N, 3), normals: (N, 3)
        N = x.shape[0]
        knn_idx = get_knn_idx(x, x, self.k + 1)[:, 1:]  # (N, k)
        
        curvatures = []
        for i in range(N):
            neighbor_normals = normals[knn_idx[i]]  # (k, 3)
            
            # 法向变化（主曲率的代理）
            normal_var = jt.std(neighbor_normals, dim=0)  # (3,)
            curvature = jt.norm(normal_var)
            curvatures.append(curvature)
        
        curvatures = jt.stack(curvatures, dim=0)  # (N,)
        return curvatures
    
    def _compute_curvature_3d(self, x, normals):
        B, N, _ = x.shape
        x_flat = x.reshape(B * N, 3)
        normals_flat = normals.reshape(B * N, 3)
        
        knn_idx = get_knn_idx(x_flat, x_flat, self.k + 1)[:, 1:]  # (B*N, k)
        
        curvatures_flat = []
        for i in range(B * N):
            neighbor_normals = normals_flat[knn_idx[i]]  # (k, 3)
            normal_var = jt.std(neighbor_normals, dim=0)
            curvature = jt.norm(normal_var)
            curvatures_flat.append(curvature)
        
        curvatures_flat = jt.stack(curvatures_flat, dim=0)  # (B*N,)
        curvatures = curvatures_flat.reshape(B, N)
        return curvatures


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取（PointNet++风格）"""
    
    def __init__(self, input_dim: int = 3, embedding_dim: int = 256, scales: list = None):
        super().__init__()
        if scales is None:
            scales = [8, 16, 32]
        
        self.scales = scales
        self.embedding_dim = embedding_dim
        
        # 为每个尺度创建MLP
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, embedding_dim // len(scales)),
                nn.ReLU(),
                nn.Linear(embedding_dim // len(scales), embedding_dim // len(scales))
            )
            for _ in scales
        ])
    
    def execute(self, x):
        """
        x: (B, N, 3) 或 (N, 3)
        返回: (B, N, embedding_dim) 或 (N, embedding_dim)
        """
        features = []
        
        for i, k in enumerate(self.scales):
            if x.ndim == 2:
                knn_idx = get_knn_idx(x, x, min(k, x.shape[0]))  # (N, k)
                knn_features = x[knn_idx]  # (N, k, 3)
                knn_features = knn_features.mean(dim=1)  # (N, 3) 全局平均
            else:
                B, N, _ = x.shape
                # 对每个batch计算多尺度
                x_flat = x.reshape(B * N, 3)
                knn_idx = get_knn_idx(x_flat, x_flat, min(k, x_flat.shape[0]))
                knn_features = x_flat[knn_idx].mean(dim=1)  # (B*N, 3)
                knn_features = knn_features.reshape(B, N, 3)
            
            feat = self.mlps[i](knn_features)
            features.append(feat)
        
        # 连接所有尺度特征
        multi_scale_feat = jt.concat(features, dim=-1)
        return multi_scale_feat


class TransformerAttention(nn.Module):
    """点云Transformer注意力模块"""
    
    def __init__(self, embedding_dim: int = 256, num_heads: int = 8, 
                 num_layers: int = 4, ff_dim: int = 1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.attention_layers = nn.ModuleList([
            nn.MultiHeadAttention(embedding_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(embedding_dim)
            for _ in range(2 * num_layers)
        ])
        
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, embedding_dim)
            )
            for _ in range(num_layers)
        ])
    
    def execute(self, x):
        """
        x: (B, N, C) 或 (N, C)
        返回: (B, N, C) 或 (N, C)
        """
        for i in range(self.num_layers):
            # 自注意力
            attn_out, _ = self.attention_layers[i](x, x, x)
            x = x + attn_out
            x = self.norm_layers[2*i](x)
            
            # 前馈网络
            ff_out = self.ff_layers[i](x)
            x = x + ff_out
            x = self.norm_layers[2*i + 1](x)
        
        return x


class GraphConvolutionLayer(nn.Module):
    """改进的图卷积层（带残差连接）"""
    
    def __init__(self, in_channels: int, out_channels: int, k: int = 16):
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # MLP用于边特征
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # 残差投影
        if in_channels != out_channels:
            self.projection = nn.Linear(in_channels, out_channels)
        else:
            self.projection = None
    
    def execute(self, x):
        """
        x: (B, N, C) 或 (N, C)
        """
        # 获取KNN
        if x.ndim == 2:
            knn_idx = get_knn_idx(x, x, self.k + 1)[:, 1:]  # (N, k)
        else:
            B, N, _ = x.shape
            x_flat = x.reshape(B * N, -1)
            knn_idx = get_knn_idx(x_flat, x_flat, self.k + 1)[:, 1:]  # (B*N, k)
        
        # 聚合邻居特征
        if x.ndim == 2:
            N = x.shape[0]
            neighbor_feats = x[knn_idx]  # (N, k, C)
            x_expanded = x.unsqueeze(1)  # (N, 1, C)
            edge_feats = jt.concat([x_expanded.repeat(1, self.k, 1), neighbor_feats], dim=-1)  # (N, k, 2C)
            
            edge_out = self.edge_mlp(edge_feats)  # (N, k, C)
            aggregated = edge_out.mean(dim=1)  # (N, C)
        else:
            B, N, C = x.shape
            x_flat = x.reshape(B * N, C)
            neighbor_feats = x_flat[knn_idx]  # (B*N, k, C)
            x_expanded = x_flat.unsqueeze(1)  # (B*N, 1, C)
            edge_feats = jt.concat([x_expanded.repeat(1, self.k, 1), neighbor_feats], dim=-1)
            
            edge_out = self.edge_mlp(edge_feats)
            aggregated = edge_out.mean(dim=1)  # (B*N, C)
            aggregated = aggregated.reshape(B, N, -1)
        
        # 残差连接
        if self.projection is not None:
            x = self.projection(x)
        
        return x + aggregated


class EnhancedFeatureExtraction(nn.Module):
    """增强的特征提取：多方法融合"""
    
    def __init__(self, k: int = 16, input_dim: int = 3, 
                 embedding_dim: int = 512, use_transformer: bool = True,
                 use_normals: bool = True, use_curvature: bool = True):
        super().__init__()
        
        self.k = k
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.use_transformer = use_transformer
        self.use_normals = use_normals
        self.use_curvature = use_curvature
        
        # 法向和曲率估计
        if use_normals:
            self.normal_estimator = NormalEstimator(k=k)
        if use_curvature:
            self.curvature_estimator = CurvatureEstimator(k=k)
        
        # 多尺度特征
        self.multi_scale = MultiScaleFeatureExtractor(
            input_dim=input_dim,
            embedding_dim=embedding_dim // 2,
            scales=[8, 16, 32]
        )
        
        # 图卷积层
        self.gc_layers = nn.ModuleList([
            GraphConvolutionLayer(embedding_dim // 4, embedding_dim // 4, k=k)
            for _ in range(3)
        ])
        
        # Transformer注意力
        if use_transformer:
            self.transformer = TransformerAttention(
                embedding_dim=embedding_dim,
                num_heads=8,
                num_layers=3,
                ff_dim=embedding_dim * 2
            )
        
        # 特征融合MLP
        feat_dim = embedding_dim // 2 + (1 if use_curvature else 0)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def execute(self, x):
        """
        x: (B, N, 3) 或 (N, 3)
        返回: (B, N, embedding_dim) 或 (N, embedding_dim)
        """
        # 多尺度特征
        multi_scale_feat = self.multi_scale(x)  # (..., embedding_dim//2)
        
        # 法向和曲率特征
        if self.use_normals:
            normals = self.normal_estimator(x)
        
        if self.use_curvature:
            curvature = self.curvature_estimator(x, normals if self.use_normals else None)
            # 扩展曲率维度用于连接
            if x.ndim == 2:
                curvature = curvature.unsqueeze(-1)  # (N, 1)
            else:
                curvature = curvature.unsqueeze(-1)  # (B, N, 1)
            
            # 融合多尺度和曲率
            features = jt.concat([multi_scale_feat, curvature], dim=-1)
        else:
            features = multi_scale_feat
        
        # 应用融合MLP
        features = self.fusion_mlp(features)  # (..., embedding_dim)
        
        # 应用Transformer（可选）
        if self.use_transformer:
            features = self.transformer(features)
        
        return features
