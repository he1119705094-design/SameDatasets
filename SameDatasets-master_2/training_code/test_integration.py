"""
集成测试脚本 - 验证新的图像块打乱和模型是否能正常工作
"""

import sys
import os
import torch
import argparse

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_transforms import ImagePatchShuffle
from networks.new_models import CombinedDetector, create_combined_detector
from utils.patch_training import PatchTrainingModel

def test_patch_shuffle():
    """测试图像块打乱功能"""
    print("=" * 50)
    print("测试图像块打乱功能")
    print("=" * 50)
    
    # 创建测试数据
    test_tensor = torch.randn(3, 224, 224)
    print(f"输入tensor形状: {test_tensor.shape}")
    
    # 创建打乱变换
    shuffle_transform = ImagePatchShuffle(patch_size=16, shuffle_prob=1.0)
    
    # 应用变换
    shuffled_tensor = shuffle_transform(test_tensor)
    print(f"打乱后tensor形状: {shuffled_tensor.shape}")
    
    # 验证形状是否相同
    assert test_tensor.shape == shuffled_tensor.shape, "形状应该保持不变"
    print("✓ 图像块打乱功能测试通过")
    
    return True

def test_combined_detector():
    """测试CombinedDetector模型"""
    print("=" * 50)
    print("测试CombinedDetector模型")
    print("=" * 50)
    
    # 创建测试数据
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 224, 224)
    print(f"输入数据形状: {test_input.shape}")
    
    # 创建模型
    model = create_combined_detector(
        img_size=224,
        patch_size=16,
        num_classes=2,
        model_size='small'
    )
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 前向传播
    with torch.no_grad():
        output = model(test_input)
        print(f"模型输出形状: {output.shape}")
    
    # 验证输出形状
    assert output.shape == (batch_size, 2), f"期望输出形状 ({batch_size}, 2)，实际得到 {output.shape}"
    print("✓ CombinedDetector模型测试通过")
    
    return True

def test_patch_training_model():
    """测试PatchTrainingModel"""
    print("=" * 50)
    print("测试PatchTrainingModel")
    print("=" * 50)
    
    # 创建模拟参数
    class MockOpt:
        def __init__(self):
            self.use_patch_model = True
            self.model_size = 'small'
            self.patch_size = 16
            self.resize_size = 224
            self.start_fresh = True
            self.use_leaky = False
            self.ckpt = None
            self.use_proj = False
            self.proj_ratio = None
            self.final_dropout = 0.5
            self.fix_backbone = False
            self.unfreeze_last_k = 0
            self.optim = 'adam'
            self.lr = 0.001
            self.beta1 = 0.9
            self.weight_decay = 0.0
            self.pretrain = None
            self.continue_epoch = None
            self.no_cuda = True  # 使用CPU进行测试
            self.checkpoints_dir = './test_checkpoints'
            self.stay_positive = None
    
    opt = MockOpt()
    
    # 创建训练模型
    training_model = PatchTrainingModel(opt, subdir='test')
    print("✓ PatchTrainingModel创建成功")
    
    # 测试前向传播
    test_input = torch.randn(2, 3, 224, 224)
    test_data = {
        'img': test_input,
        'target': torch.tensor([0, 1]),
        'path': ['test1.jpg', 'test2.jpg']
    }
    
    # 测试训练
    training_model.model.eval()
    with torch.no_grad():
        if opt.use_patch_model:
            output = training_model.model(test_input)
        else:
            output, _ = training_model.model(test_input)
        print(f"训练模型输出形状: {output.shape}")
    
    print("✓ PatchTrainingModel测试通过")
    return True

def test_data_processing_integration():
    """测试数据处理集成"""
    print("=" * 50)
    print("测试数据处理集成")
    print("=" * 50)
    
    # 模拟完整的处理流程
    from utils.processing import make_processing
    
    class MockOpt:
        def __init__(self):
            self.patch_size = 16
            self.resize_size = 224
            self.crop_size = 224
            self.load_size = -1
            self.loadSize = -1  # 添加这个属性
            self.resizeSize = -1
            self.cropSize = 224
            self.no_random_crop = False
            self.resize_prob = 0.0
            self.jitter_prob = 0.0
            self.colordist_prob = 0.0
            self.cutout_prob = 0.0
            self.noise_prob = 0.0
            self.blur_prob = 0.0
            self.cmp_prob = 0.0
            self.rot90_prob = 0.0
            self.no_flip = False
            self.hpf_prob = 0.0
            self.pre_crop_prob = 0.0
            self.rz_interp = "bilinear"
            self.blur_sig = [0.5]
            self.cmp_method = ["cv2"]
            self.cmp_qual = [75]
            self.resize_ratio = 1.0
            self.norm_type = "resnet"
            self.num_views = 0
            self.flex_rz = False
    
    opt = MockOpt()
    
    # 创建处理流程
    transform = make_processing(opt)
    print("✓ 数据处理流程创建成功")
    
    # 测试处理
    from PIL import Image
    import numpy as np
    
    # 创建测试图像
    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    processed = transform(test_image)
    
    print(f"处理后图像形状: {processed.shape}")
    print("✓ 数据处理集成测试通过")
    
    return True

def main():
    """主测试函数"""
    print("开始集成测试...")
    print()
    
    try:
        # 运行所有测试
        test_patch_shuffle()
        print()
        
        test_combined_detector()
        print()
        
        test_patch_training_model()
        print()
        
        test_data_processing_integration()
        print()
        
        print("=" * 50)
        print("🎉 所有测试通过！集成成功！")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
