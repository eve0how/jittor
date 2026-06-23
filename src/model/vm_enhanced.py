"""
增强的点云降噪模型
包含：多种损失函数、自适应噪声、迭代降噪、P2S约束
"""
from math import ceil
from typing import Dict, List, Optional, Tuple

import jittor as jt
import numpy as np

from .feature_enhanced import EnhancedFeatureExtraction
from .feature import Decoder
from .spec import ModelSpec
from ..data.asset import Asset


def get_random_indices(n, m):
    assert m < n
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()


class AdaptiveNoiseEstimator(jt.nn.Module):
    """自适应噪声强度估计器"""
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.noise_predictor = jt.nn.Sequential(
            jt.nn.Linear(embedding_dim, embedding_dim // 2),
            jt.nn.ReLU(),
            jt.nn.Linear(embedding_dim // 2, 1),
            jt.nn.Softplus()  # 确保输出为正
        )
    
    def execute(self, features):
        """
        features: (..., embedding_dim)
        返回: (..., 1) 的噪声强度
        """
        noise_level = self.noise_predictor(features)
        return noise_level + 1e-6  # 避免除零


class EnhancedVelocityModule(ModelSpec):
    """增强的速度模块（位移预测）"""
    
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        
        cfg = self.model_config
        
        # 几何参数
        self.frame_knn = cfg.get('frame_knn', 16)
        self.num_train_points = cfg.get('num_train_points', 128)
        
        # 损失函数参数
        self.dsm_sigma = cfg.get('dsm_sigma', 0.01)
        self.loss_type = cfg.get('loss_type', 'supervised')  # supervised/hybrid/diffusion
        self.use_p2s = cfg.get('use_p2s', False)
        self.use_adaptive_noise = cfg.get('use_adaptive_noise', True)
        self.lambda_p2s = cfg.get('lambda_p2s', 0.1)
        
        # 推理参数
        self.num_inference_steps = cfg.get('num_inference_steps', 4)
        self.use_iterative_refinement = cfg.get('use_iterative_refinement', False)
        
        # 增强特征提取
        self.encoder = EnhancedFeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=cfg.get('feat_embedding_dim', 256),
            use_transformer=cfg.get('use_transformer', True),
            use_normals=cfg.get('use_normals', True),
            use_curvature=cfg.get('use_curvature', True)
        )
        
        # 解码器
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg.get('decoder_hidden_dim', 64),
        )
        
        # 自适应噪声估计
        if self.use_adaptive_noise:
            self.noise_estimator = AdaptiveNoiseEstimator(
                embedding_dim=self.encoder.embedding_dim
            )
    
    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean, curvature_weights=None):
        """
        监督学习损失
        pc_noisy: (B, N, 3) - 原始含噪点
        pc_mix: (B, N, 3) - 混合点（用于特征提取）
        pc_clean: (B, N, 3) - 干净点
        curvature_weights: (B, N) - 曲率权重（可选）
        """
        B, N_mix, d = pc_mix.shape
        
        # 采样训练点
        pnt_idx = get_random_indices(N_mix, self.num_train_points)
        
        # 特征提取
        feat = self.encoder(pc_mix)  # (B, N, F)
        F_dim = feat.shape[2]
        
        # 采样
        feat_sampled = feat[:, pnt_idx, :]
        pc_noisy_sampled = pc_noisy[:, pnt_idx, :]
        pc_mix_sampled = pc_mix[:, pnt_idx, :]
        pc_clean_sampled = pc_clean[:, pnt_idx, :]
        
        # 目标：位移向量
        displacement_target = pc_clean_sampled - pc_noisy_sampled  # (B, M, 3)
        
        # 预测位移
        pred_displacement = self.decoder(
            c=feat_sampled.reshape(-1, F_dim)
        ).reshape(B, len(pnt_idx), d)
        
        # 基础监督损失
        if self.use_adaptive_noise:
            # 自适应噪声加权
            noise_level = self.noise_estimator(feat_sampled)  # (B, M, 1)
            supervised_loss = (
                ((pred_displacement - displacement_target) ** 2) / 
                (noise_level ** 2 + 1e-6)
            ).sum(dim=-1).mean()
        else:
            supervised_loss = (
                ((pred_displacement - displacement_target) ** 2) / self.dsm_sigma
            ).sum(dim=-1).mean()
        
        # 曲率加权（保留细节）
        if curvature_weights is not None:
            curvature_w = curvature_weights[:, pnt_idx]  # (B, M)
            # 高曲率点给予更高权重
            curvature_w = 1.0 + curvature_w  # 范围 [1, 2]
            supervised_loss = (supervised_loss * curvature_w.mean()).mean()
        
        return supervised_loss
    
    def get_consistency_loss(self, pc_noisy, pc_mix, pc_clean):
        """
        一致性损失：预测的降噪点应该在真实表面附近
        """
        B, N_mix, d = pc_mix.shape
        pnt_idx = get_random_indices(N_mix, self.num_train_points)
        
        feat = self.encoder(pc_mix)
        F_dim = feat.shape[2]
        
        feat_sampled = feat[:, pnt_idx, :]
        pc_noisy_sampled = pc_noisy[:, pnt_idx, :]
        pc_clean_sampled = pc_clean[:, pnt_idx, :]
        
        pred_displacement = self.decoder(
            c=feat_sampled.reshape(-1, F_dim)
        ).reshape(B, len(pnt_idx), d)
        
        # 降噪后的点
        pc_denoised = pc_noisy_sampled + pred_displacement
        
        # 一致性：降噪点应该接近真实点
        consistency_loss = jt.norm(pc_denoised - pc_clean_sampled, dim=-1).mean()
        
        return consistency_loss
    
    def training_step(self, batch: Dict) -> Dict:
        """训练步骤"""
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        
        # 计算曲率权重（可选）
        curvature_weights = None
        if self.model_config.get('use_curvature', True):
            from .feature_enhanced import CurvatureEstimator, NormalEstimator
            normal_est = NormalEstimator(k=self.frame_knn)
            normals = normal_est(pc_mix)
            curvature_est = CurvatureEstimator(k=self.frame_knn)
            curvature_weights = curvature_est(pc_mix, normals)
        
        # 主损失
        loss = self.get_supervised_loss(pc_noisy, pc_mix, pc_clean, curvature_weights)
        
        # 混合损失
        if self.loss_type == 'hybrid':
            consistency_loss = self.get_consistency_loss(pc_noisy, pc_mix, pc_clean)
            loss = loss + 0.1 * consistency_loss
        
        return {"loss": loss}
    
    def denoise_single_step(self, pcl):
        """单步降噪"""
        with jt.no_grad():
            feat = self.encoder(pcl)
            F_dim = feat.shape[2]
            B, N, d = pcl.shape
            
            pred_displacement = self.decoder(
                c=feat.reshape(-1, F_dim)
            ).reshape(B, N, d)
            
            pcl_denoised = pcl + pred_displacement
        
        return pcl_denoised
    
    def denoise_iterative(self, pcl_noisy, num_steps: int = 4):
        """
        迭代降噪（支持多步细化）
        pcl_noisy: (B, N, 3)
        """
        B, N, d = pcl_noisy.shape
        pcl = pcl_noisy.clone()
        
        with jt.no_grad():
            for step in range(num_steps):
                feat = self.encoder(pcl)
                F_dim = feat.shape[2]
                
                pred_displacement = self.decoder(
                    c=feat.reshape(-1, F_dim)
                ).reshape(B, N, d)
                
                # 步长衰减
                step_size = 1.0 / (step + 1)
                pcl = pcl + step_size * pred_displacement
        
        return pcl, None
    
    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int = 4):
        """
        Langevin动力学（添加随机性以避免过度平滑）
        """
        B, N, d = pcl_noisy.shape
        
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            
            for it in range(num_steps):
                feat = self.encoder(pcl_next)
                F_dim = feat.shape[2]
                
                pred_dir = self.decoder(
                    c=feat.reshape(-1, F_dim)
                ).reshape(B, N, d)
                
                # Langevin更新：添加噪声避免模式崩塌
                temperature = 0.01 * (1 - it / num_steps)  # 退火
                noise = jt.randn_like(pcl_next) * jt.sqrt(jt.array(temperature))
                
                pcl_next = pcl_next + (1.0 / num_steps) * pred_dir + noise
        
        return pcl_next, None
    
    def validation_step(self, batch: Dict) -> Dict:
        """验证步骤"""
        pc_noisy = batch['pc_noisy']
        pc_clean = batch['pc_clean']
        
        # 推理
        if self.use_iterative_refinement:
            pc_pred, _ = self.denoise_iterative(pc_noisy, num_steps=self.num_inference_steps)
        else:
            pc_pred, _ = self.denoise_langevin_dynamics(pc_noisy, num_steps=self.num_inference_steps)
        
        # 计算误差
        mse = jt.mean((pc_pred - pc_clean) ** 2)
        rmse = jt.sqrt(mse)
        
        return {
            "val_mse": mse,
            "val_rmse": rmse,
            "pc_pred": pc_pred.detach(),
        }
    
    def predict_step(self, batch: Dict) -> Dict:
        """预测步骤（推理）"""
        pc_noisy = batch['pc_noisy']
        
        # 使用迭代优化
        if self.use_iterative_refinement:
            pc_pred, _ = self.denoise_iterative(pc_noisy, num_steps=self.num_inference_steps)
        else:
            pc_pred, _ = self.denoise_langevin_dynamics(pc_noisy, num_steps=self.num_inference_steps)
        
        return {"pc_pred": pc_pred}
    
    def get_forward_features(self, x):
        """获取特征（用于可视化或分析）"""
        return self.encoder(x)


class DiffusionDenoiser(ModelSpec):
    """基于扩散过程的点云降噪模型（高级）"""
    
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        
        cfg = self.model_config
        self.num_diffusion_steps = cfg.get('num_diffusion_steps', 50)
        self.noise_schedule = cfg.get('noise_schedule', 'linear')
        
        # 特征提取器
        self.encoder = EnhancedFeatureExtraction(
            k=cfg.get('frame_knn', 16),
            input_dim=3,
            embedding_dim=cfg.get('feat_embedding_dim', 256),
            use_transformer=True,
            use_normals=True,
            use_curvature=True
        )
        
        # 得分网络
        self.score_decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg.get('decoder_hidden_dim', 64),
        )
        
        # 时间编码
        self.time_embedding = jt.nn.Sequential(
            jt.nn.Linear(1, 64),
            jt.nn.ReLU(),
            jt.nn.Linear(64, self.encoder.embedding_dim)
        )
        
        # 构建噪声时间表
        self._build_noise_schedule()
    
    def _build_noise_schedule(self):
        """构建β时间表"""
        if self.noise_schedule == 'linear':
            self.betas = jt.linspace(0.0001, 0.02, self.num_diffusion_steps)
        elif self.noise_schedule == 'quadratic':
            self.betas = jt.linspace(0.0001 ** 0.5, 0.02 ** 0.5, self.num_diffusion_steps) ** 2
        
        self.alphas = 1.0 - self.betas
        self.alpha_bars = jt.cumprod(self.alphas, dim=0)
    
    def training_step(self, batch: Dict) -> Dict:
        """扩散模型训练"""
        pc_noisy = batch['pc_noisy']
        pc_clean = batch['pc_clean']
        B, N, _ = pc_noisy.shape
        
        # 随机时间步
        t = jt.randint(0, self.num_diffusion_steps, (B,))
        
        # 前向扩散：q(x_t | x_0)
        alpha_bar_t = self.alpha_bars[t].reshape(B, 1, 1)
        
        # 从干净点云添加噪声到噪声点云
        noise = jt.randn_like(pc_clean)
        x_t = jt.sqrt(alpha_bar_t) * pc_clean + jt.sqrt(1 - alpha_bar_t) * noise
        
        # 预测得分/噪声
        time_emb = self.time_embedding(t.float().unsqueeze(1))  # (B, D)
        feat = self.encoder(x_t)  # (B, N, D)
        feat = feat + time_emb.unsqueeze(1)  # 广播加法
        
        pred_score = self.score_decoder(feat.reshape(-1, feat.shape[-1]))
        pred_score = pred_score.reshape(B, N, 3)
        
        # 得分匹配损失
        score_loss = jt.mean((pred_score - noise) ** 2)
        
        return {"loss": score_loss}
    
    def denoise_reverse_diffusion(self, x_noisy, num_steps: int = 50):
        """反向扩散过程"""
        with jt.no_grad():
            x_t = x_noisy.clone()
            
            for t in range(num_steps - 1, -1, -1):
                t_tensor = jt.array([t]).long()
                beta_t = self.betas[t]
                alpha_t = self.alphas[t]
                alpha_bar_t = self.alpha_bars[t]
                
                # 预测得分
                time_emb = self.time_embedding(jt.array([t]).float().unsqueeze(1))
                feat = self.encoder(x_t)
                feat = feat + time_emb.unsqueeze(1)
                
                pred_score = self.score_decoder(feat.reshape(-1, feat.shape[-1]))
                pred_score = pred_score.reshape(x_t.shape)
                
                # Langevin动力学更新
                z = jt.randn_like(x_t)
                x_t = (1 / jt.sqrt(alpha_t)) * (x_t - (beta_t / jt.sqrt(1 - alpha_bar_t)) * pred_score)
                x_t = x_t + jt.sqrt(beta_t) * z
        
        return x_t, None
