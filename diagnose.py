#!/usr/bin/env python3
"""
快速诊断脚本 - 检查环境和验证新代码
"""

import os
import sys
import traceback

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def check_imports():
    """检查所需的库"""
    print_header("检查导入")
    
    libraries = {
        'jittor': 'import jittor as jt',
        'numpy': 'import numpy as np',
        'omegaconf': 'from omegaconf import OmegaConf',
        'scipy': 'import scipy',
        'trimesh': 'import trimesh',
    }
    
    passed = 0
    for lib, import_stmt in libraries.items():
        try:
            exec(import_stmt)
            print(f"✓ {lib:15s} OK")
            passed += 1
        except ImportError as e:
            print(f"✗ {lib:15s} FAILED: {e}")
    
    print(f"\n总计: {passed}/{len(libraries)} 通过")
    return passed == len(libraries)

def check_new_modules():
    """检查新增模块是否可导入"""
    print_header("检查新增模块")
    
    modules = [
        ('src.model.feature_enhanced', 'EnhancedFeatureExtraction'),
        ('src.model.vm_enhanced', 'EnhancedVelocityModule'),
        ('src.model.vm_enhanced', 'DiffusionDenoiser'),
    ]
    
    passed = 0
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
            passed += 1
        except (ImportError, AttributeError) as e:
            print(f"✗ {module_name}.{class_name}: {e}")
    
    print(f"\n总计: {passed}/{len(modules)} 通过")
    return passed == len(modules)

def test_feature_extraction():
    """测试特征提取模块"""
    print_header("测试特征提取")
    
    try:
        import jittor as jt
        from src.model.feature_enhanced import (
            NormalEstimator, 
            CurvatureEstimator,
            MultiScaleFeatureExtractor,
            EnhancedFeatureExtraction
        )
        
        # 生成随机点云
        points = jt.randn(2, 1000, 3)
        print(f"输入点云: {points.shape}")
        
        # 测试法向估计
        normal_est = NormalEstimator(k=16)
        normals = normal_est(points)
        print(f"✓ 法向估计: {normals.shape}")
        
        # 测试曲率估计
        curvature_est = CurvatureEstimator(k=16)
        curvature = curvature_est(points, normals)
        print(f"✓ 曲率估计: {curvature.shape}")
        
        # 测试多尺度特征
        multi_scale = MultiScaleFeatureExtractor(
            input_dim=3,
            embedding_dim=256,
            scales=[8, 16, 32]
        )
        feat = multi_scale(points)
        print(f"✓ 多尺度特征: {feat.shape}")
        
        # 测试增强特征提取
        enhanced = EnhancedFeatureExtraction(
            k=16,
            input_dim=3,
            embedding_dim=256,
            use_transformer=True,
            use_normals=True,
            use_curvature=True
        )
        enhanced_feat = enhanced(points)
        print(f"✓ 增强特征提取: {enhanced_feat.shape}")
        
        print("\n✅ 所有特征提取模块测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ 特征提取模块测试失败:")
        traceback.print_exc()
        return False

def test_models():
    """测试模型"""
    print_header("测试模型")
    
    try:
        import jittor as jt
        from src.model.vm_enhanced import EnhancedVelocityModule
        
        # 构造配置
        model_config = {
            'frame_knn': 16,
            'num_train_points': 128,
            'feat_embedding_dim': 256,
            'decoder_hidden_dim': 64,
            'dsm_sigma': 0.01,
            'use_normals': True,
            'use_curvature': True,
            'use_transformer': False,
            'use_adaptive_noise': False,
        }
        
        transform_config = {}
        
        # 创建模型
        model = EnhancedVelocityModule(model_config, transform_config)
        print(f"✓ 模型创建成功")
        
        # 测试前向传播
        pc_noisy = jt.randn(2, 5000, 3)
        pc_mix = pc_noisy.clone()
        pc_clean = pc_noisy + 0.01 * jt.randn(2, 5000, 3)
        
        batch = {
            'pc_noisy': pc_noisy,
            'pc_mix': pc_mix,
            'pc_clean': pc_clean,
        }
        
        result = model.training_step(batch)
        print(f"✓ 训练步通过, loss: {result['loss'].numpy():.6f}")
        
        # 测试推理
        pc_denoised, _ = model.denoise_langevin_dynamics(pc_noisy, num_steps=2)
        print(f"✓ 推理步通过, 输出shape: {pc_denoised.shape}")
        
        print("\n✅ 所有模型测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ 模型测试失败:")
        traceback.print_exc()
        return False

def check_config_files():
    """检查配置文件"""
    print_header("检查配置文件")
    
    config_files = [
        'configs/model/vm_enhanced_stage1.yaml',
        'configs/model/vm_enhanced_stage2.yaml',
        'configs/model/vm_enhanced_stage3.yaml',
        'configs/task/train_vm_enhanced_stage1.yaml',
        'configs/task/train_vm_enhanced_stage2.yaml',
        'configs/task/predict_vm_enhanced.yaml',
    ]
    
    passed = 0
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ {config_file}")
            passed += 1
        else:
            print(f"✗ {config_file} (不存在)")
    
    print(f"\n总计: {passed}/{len(config_files)} 存在")
    return passed == len(config_files)

def check_documentation():
    """检查文档文件"""
    print_header("检查文档")
    
    doc_files = [
        'IMPROVEMENTS_GUIDE.md',
        'QUICK_START.md',
        'INTEGRATION_GUIDE.md',
        'COMPLETE_PLAN.md',
    ]
    
    passed = 0
    for doc_file in doc_files:
        if os.path.exists(doc_file):
            size_kb = os.path.getsize(doc_file) / 1024
            print(f"✓ {doc_file:30s} ({size_kb:.1f} KB)")
            passed += 1
        else:
            print(f"✗ {doc_file} (不存在)")
    
    print(f"\n总计: {passed}/{len(doc_files)} 存在")
    return passed == len(doc_files)

def print_summary(results):
    """打印总结"""
    print_header("诊断总结")
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    print("检查项目:")
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    print(f"\n总体: {passed_checks}/{total_checks} 通过")
    
    if passed_checks == total_checks:
        print("\n🎉 所有检查通过！可以开始训练了！")
        print("\n建议从以下命令开始:")
        print("  python run.py --task configs/task/train_vm_enhanced_stage1.yaml")
    else:
        print("\n⚠️  有部分检查失败，请修复后重试")
    
    return passed_checks == total_checks

def main():
    print("\n" + "="*60)
    print("  点云降噪 - 增强方案诊断工具")
    print("="*60)
    
    results = {
        "库导入检查": check_imports(),
        "新增模块检查": check_new_modules(),
        "特征提取测试": test_feature_extraction(),
        "模型测试": test_models(),
        "配置文件检查": check_config_files(),
        "文档文件检查": check_documentation(),
    }
    
    success = print_summary(results)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
