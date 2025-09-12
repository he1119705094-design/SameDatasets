"""
图像块打乱变换类 - 基于 FakeImageDetection 的核心思想
通过破坏图像的全局语义信息，迫使检测器关注局部"生成器伪影"

这个模块实现了 ImagePatchShuffle 类，用于在数据加载时对图像进行图像块打乱操作。
"""

import torch
import torch.nn as nn
import random
import numpy as np
from typing import Tuple, Optional


class ImagePatchShuffle:
    """
    图像块打乱变换类
    
    核心思想：
    1. 将输入图像分割成固定大小的图像块
    2. 随机打乱这些图像块的顺序
    3. 重新组合成与原图相同尺寸的图像
    
    这样做可以破坏图像的全局语义信息，迫使检测器关注局部特征和生成器伪影。
    """
    
    def __init__(self, patch_size: int = 16, shuffle_prob: float = 1.0):
        """
        初始化图像块打乱变换
        
        Args:
            patch_size (int): 图像块的大小，默认16x16像素
            shuffle_prob (float): 执行打乱操作的概率，默认1.0（总是打乱）
        """
        self.patch_size = patch_size
        self.shuffle_prob = shuffle_prob
        
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        对输入的tensor进行图像块打乱
        
        Args:
            tensor (torch.Tensor): 输入的图像tensor，形状为 (C, H, W)
            
        Returns:
            torch.Tensor: 打乱后的图像tensor，形状与原tensor相同
        """
        # 检查是否需要执行打乱操作
        if random.random() > self.shuffle_prob:
            return tensor
            
        # 获取tensor的维度信息
        if len(tensor.shape) == 4:
            # 如果是4维tensor (B, C, H, W)，取第一个样本
            B, C, H, W = tensor.shape
            tensor = tensor[0]  # 取第一个样本
        else:
            C, H, W = tensor.shape
        
        # 确保图像尺寸能被patch_size整除
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            # 如果不能整除，先调整图像尺寸
            new_H = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size
            new_W = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size
            
            # 使用插值调整尺寸
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), 
                size=(new_H, new_W), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            H, W = new_H, new_W
        
        # 计算图像块的数量
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # 将图像分割成图像块
        patches = tensor.view(
            C, 
            num_patches_h, 
            self.patch_size, 
            num_patches_w, 
            self.patch_size
        )
        
        # 重新排列维度以便打乱
        patches = patches.permute(0, 1, 3, 2, 4).contiguous()
        patches = patches.view(C, num_patches_h * num_patches_w, self.patch_size, self.patch_size)
        
        # 生成随机打乱索引
        num_patches = num_patches_h * num_patches_w
        shuffle_indices = torch.randperm(num_patches)
        
        # 打乱图像块
        shuffled_patches = patches[:, shuffle_indices, :, :]
        
        # 重新组合成完整图像
        shuffled_patches = shuffled_patches.view(C, num_patches_h, num_patches_w, self.patch_size, self.patch_size)
        shuffled_patches = shuffled_patches.permute(0, 1, 3, 2, 4).contiguous()
        
        # 恢复原始尺寸
        shuffled_tensor = shuffled_patches.view(C, H, W)
        
        return shuffled_tensor
    
    def __repr__(self):
        return f"ImagePatchShuffle(patch_size={self.patch_size}, shuffle_prob={self.shuffle_prob})"


class AdaptiveImagePatchShuffle:
    """
    自适应图像块打乱变换类
    
    根据图像尺寸自动调整patch_size，确保打乱效果的一致性
    """
    
    def __init__(self, base_patch_size: int = 16, shuffle_prob: float = 1.0):
        """
        初始化自适应图像块打乱变换
        
        Args:
            base_patch_size (int): 基础图像块大小
            shuffle_prob (float): 执行打乱操作的概率
        """
        self.base_patch_size = base_patch_size
        self.shuffle_prob = shuffle_prob
        
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        对输入的tensor进行自适应图像块打乱
        
        Args:
            tensor (torch.Tensor): 输入的图像tensor，形状为 (C, H, W)
            
        Returns:
            torch.Tensor: 打乱后的图像tensor
        """
        if random.random() > self.shuffle_prob:
            return tensor
            
        C, H, W = tensor.shape
        
        # 根据图像尺寸自适应调整patch_size
        # 确保图像被分割成合理数量的图像块（4-16个）
        min_dim = min(H, W)
        if min_dim <= 64:
            patch_size = min_dim // 4
        elif min_dim <= 128:
            patch_size = min_dim // 8
        else:
            patch_size = self.base_patch_size
            
        # 确保patch_size是合理的
        patch_size = max(8, min(patch_size, min_dim // 2))
        
        # 使用ImagePatchShuffle进行打乱
        shuffle_transform = ImagePatchShuffle(patch_size=patch_size, shuffle_prob=1.0)
        return shuffle_transform(tensor)
    
    def __repr__(self):
        return f"AdaptiveImagePatchShuffle(base_patch_size={self.base_patch_size}, shuffle_prob={self.shuffle_prob})"


def create_patch_shuffle_transform(patch_size: int = 16, shuffle_prob: float = 1.0, adaptive: bool = False):
    """
    创建图像块打乱变换的工厂函数
    
    Args:
        patch_size (int): 图像块大小
        shuffle_prob (float): 打乱概率
        adaptive (bool): 是否使用自适应版本
        
    Returns:
        ImagePatchShuffle 或 AdaptiveImagePatchShuffle 实例
    """
    if adaptive:
        return AdaptiveImagePatchShuffle(base_patch_size=patch_size, shuffle_prob=shuffle_prob)
    else:
        return ImagePatchShuffle(patch_size=patch_size, shuffle_prob=shuffle_prob)


# 测试函数
def test_patch_shuffle():
    """测试图像块打乱功能"""
    # 创建一个测试图像tensor
    test_tensor = torch.randn(3, 64, 64)
    
    # 创建打乱变换
    shuffle_transform = ImagePatchShuffle(patch_size=16, shuffle_prob=1.0)
    
    # 应用变换
    shuffled_tensor = shuffle_transform(test_tensor)
    
    print(f"原始tensor形状: {test_tensor.shape}")
    print(f"打乱后tensor形状: {shuffled_tensor.shape}")
    print(f"形状是否相同: {test_tensor.shape == shuffled_tensor.shape}")
    
    # 测试自适应版本
    adaptive_transform = AdaptiveImagePatchShuffle(base_patch_size=16, shuffle_prob=1.0)
    adaptive_shuffled = adaptive_transform(test_tensor)
    
    print(f"自适应打乱后tensor形状: {adaptive_shuffled.shape}")


if __name__ == "__main__":
    test_patch_shuffle()
