"""
运行示例脚本 - 展示如何使用新的图像块打乱和模型进行训练

这个脚本展示了如何运行结合了FakeImageDetection和AlignedForensics核心思想的训练
"""

import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_usage():
    """打印使用说明"""
    print("=" * 80)
    print("AlignedForensics + FakeImageDetection 集成训练示例")
    print("=" * 80)
    print()
    print("这个脚本展示了如何运行结合了两篇论文核心思想的训练：")
    print("1. AlignedForensics: 使用LDM自编码器生成高度对齐的伪造图像")
    print("2. FakeImageDetection: 通过图像块打乱破坏全局语义，关注局部伪影")
    print()
    print("使用方法：")
    print()
    print("1. 使用传统ResNet模型（原有方法）：")
    print("   python train.py --name traditional_test --dataroot ../datasets --arch res50nodown --batch_size 8 --lr 0.0001")
    print()
    print("2. 使用新的图像块打乱 + ResNet模型：")
    print("   python train.py --name patch_shuffle_test --dataroot ../datasets --arch res50nodown --batch_size 8 --lr 0.0001 --patch_size 16")
    print()
    print("3. 使用新的CombinedDetector模型（推荐）：")
    print("   python train.py --name combined_detector_test --dataroot ../datasets --use_patch_model --model_size small --batch_size 8 --lr 0.0001 --patch_size 16")
    print()
    print("4. 使用CombinedDetector + batched_syncing（最佳效果）：")
    print("   python train.py --name best_test --dataroot ../datasets --use_patch_model --model_size medium --batch_size 8 --lr 0.0001 --patch_size 16 --batched_syncing --use_inversions")
    print()
    print("参数说明：")
    print("  --use_patch_model: 使用新的CombinedDetector模型")
    print("  --model_size: 模型大小 (small/medium/large)")
    print("  --patch_size: 图像块大小 (默认16)")
    print("  --batched_syncing: 启用批次同步（AlignedForensics的核心特性）")
    print("  --use_inversions: 使用LDM重建的伪造图像")
    print()
    print("注意事项：")
    print("1. 确保数据集路径正确")
    print("2. 根据GPU内存调整batch_size")
    print("3. 建议先使用small模型进行测试")
    print("4. 使用--batched_syncing和--use_inversions可以获得最佳效果")
    print()
    print("=" * 80)

def create_sample_script():
    """创建示例运行脚本"""
    script_content = '''#!/bin/bash
# 示例训练脚本

echo "开始训练..."

# 示例1: 传统ResNet
echo "训练传统ResNet模型..."
python train.py \\
    --name traditional_test \\
    --dataroot ../datasets \\
    --arch res50nodown \\
    --batch_size 8 \\
    --lr 0.0001 \\
    --num_epoches 100

# 示例2: ResNet + 图像块打乱
echo "训练ResNet + 图像块打乱模型..."
python train.py \\
    --name patch_shuffle_test \\
    --dataroot ../datasets \\
    --arch res50nodown \\
    --batch_size 8 \\
    --lr 0.0001 \\
    --patch_size 16 \\
    --num_epoches 100

# 示例3: CombinedDetector (推荐)
echo "训练CombinedDetector模型..."
python train.py \\
    --name combined_detector_test \\
    --dataroot ../datasets \\
    --use_patch_model \\
    --model_size small \\
    --batch_size 8 \\
    --lr 0.0001 \\
    --patch_size 16 \\
    --num_epoches 100

# 示例4: 最佳配置
echo "训练最佳配置模型..."
python train.py \\
    --name best_test \\
    --dataroot ../datasets \\
    --use_patch_model \\
    --model_size medium \\
    --batch_size 8 \\
    --lr 0.0001 \\
    --patch_size 16 \\
    --batched_syncing \\
    --use_inversions \\
    --num_epoches 100

echo "训练完成！"
'''
    
    with open('run_training_examples.sh', 'w') as f:
        f.write(script_content)
    
    print("已创建示例脚本: run_training_examples.sh")
    print("在Linux/Mac上可以使用: bash run_training_examples.sh")

def main():
    """主函数"""
    print_usage()
    print()
    
    # 创建示例脚本
    create_sample_script()
    print()
    
    print("现在您可以开始训练了！")
    print("建议从CombinedDetector模型开始：")
    print("python train.py --name test --dataroot ../datasets --use_patch_model --model_size small --batch_size 8 --lr 0.0001 --patch_size 16")

if __name__ == "__main__":
    main()
