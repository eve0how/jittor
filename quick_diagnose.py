#!/usr/bin/env python3
"""
快速诊断脚本 - 检查Stage1改进代码是否正确安装
运行: python quick_diagnose.py
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_files():
    """检查新文件是否存在"""
    print("📁 检查文件...")
    files = [
        "src/model/feature_enhanced_quick.py",
        "src/model/vm_stage1.py",
        "configs/model/vm_stage1_quick.yaml",
        "configs/task/train_stage1_quick.yaml",
    ]
    
    all_exist = True
    for f in files:
        exists = os.path.exists(f)
        status = "✓" if exists else "✗"
        print(f"  {status} {f}")
        all_exist = all_exist and exists
    
    return all_exist

def check_imports():
    """检查导入是否成功"""
    print("\n📚 检查导入...")
    
    try:
        from src.model.feature_enhanced_quick import (
            QuickNormalEstimator,
            QuickCurvatureEstimator,
            QuickMultiScaleFeature,
            QuickFeatureExtraction,
            PointwiseWeightedLoss
        )
        print("  ✓ feature_enhanced_quick 导入成功")
    except Exception as e:
        print(f"  ✗ feature_enhanced_quick 导入失败: {e}")
        return False
    
    try:
        from src.model.vm_stage1 import VelocityModuleStage1
        print("  ✓ vm_stage1 导入成功")
    except Exception as e:
        print(f"  ✗ vm_stage1 导入失败: {e}")
        return False
    
    try:
        from src.model.parse import get_model
        print("  ✓ parse 导入成功")
    except Exception as e:
        print(f"  ✗ parse 导入失败: {e}")
        return False
    
    return True

def check_model_registration():
    """检查模型是否正确注册"""
    print("\n🔧 检查模型注册...")
    
    try:
        from src.model.parse import get_model
        
        # 测试创建模型
        config = {
            '__target__': 'VelocityModuleStage1',
            'frame_knn': 24,
            'num_train_points': 5000,
            'feat_embedding_dim': 256,
            'decoder_hidden_dim': 128,
            'dsm_sigma': 0.01,
            'use_curvature_weighting': True,
            'num_inference_steps': 4,
        }
        
        model = get_model(config, transform_config={})
        print(f"  ✓ VelocityModuleStage1 模型创建成功")
        print(f"    类型: {type(model).__name__}")
        print(f"    特征提取器: {type(model.encoder).__name__}")
        return True
        
    except Exception as e:
        print(f"  ✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_config_files():
    """检查配置文件格式"""
    print("\n⚙️ 检查配置文件...")
    
    try:
        import yaml
        
        # 检查模型配置
        with open("configs/model/vm_stage1_quick.yaml", "r") as f:
            model_cfg = yaml.safe_load(f)
        print(f"  ✓ 模型配置加载成功")
        print(f"    目标: {model_cfg.get('__target__', 'N/A')}")
        print(f"    frame_knn: {model_cfg.get('frame_knn', 'N/A')}")
        
        # 检查任务配置
        with open("configs/task/train_stage1_quick.yaml", "r") as f:
            task_cfg = yaml.safe_load(f)
        print(f"  ✓ 任务配置加载成功")
        print(f"    batch_size: {task_cfg.get('batch_size', 'N/A')}")
        print(f"    max_epochs: {task_cfg.get('max_epochs', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 配置文件检查失败: {e}")
        return False

def main():
    """主诊断函数"""
    print("=" * 60)
    print("🚀 Stage1 快速改进 - 诊断工具")
    print("=" * 60)
    
    results = []
    
    # 运行所有检查
    results.append(("文件检查", check_files()))
    results.append(("导入检查", check_imports()))
    results.append(("模型注册", check_model_registration()))
    results.append(("配置文件", check_config_files()))
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 诊断结果:")
    print("=" * 60)
    
    all_pass = True
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status}: {name}")
        all_pass = all_pass and result
    
    print("=" * 60)
    
    if all_pass:
        print("\n✅ 所有检查通过！")
        print("\n🎯 下一步：")
        print("  python run.py --task configs/task/train_stage1_quick.yaml")
        print("\n预期性能: 75分 (相比baseline 60分)")
        return 0
    else:
        print("\n❌ 有些检查失败，请检查上面的错误")
        print("\n提示：")
        print("  1. 确保所有新文件都已创建")
        print("  2. 确保parse.py已修改（添加VelocityModuleStage1导入）")
        print("  3. 查看具体错误信息进行排查")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
