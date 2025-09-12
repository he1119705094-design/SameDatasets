"""
基于图像块的检测器模型 - 结合 FakeImageDetection 和 AlignedForensics 的核心思想

这个模块实现了：
1. PatchFeatureExtractor: 用于从图像块中提取特征
2. CombinedDetector: 结合图像块特征提取和全局特征的综合检测器

核心思想：
- 通过图像块打乱破坏全局语义，迫使模型关注局部生成器伪影
- 使用Transformer架构处理图像块序列
- 结合局部和全局特征进行最终分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
import numpy as np


class PatchEmbedding(nn.Module):
    """
    图像块嵌入层 - 将图像块转换为特征向量
    
    将输入图像分割成patches，然后通过线性投影将每个patch转换为特征向量
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        """
        初始化图像块嵌入层
        
        Args:
            img_size (int): 输入图像尺寸
            patch_size (int): 图像块尺寸
            in_channels (int): 输入通道数
            embed_dim (int): 嵌入维度
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 线性投影层：将patch转换为特征向量
        self.projection = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像，形状为 (B, C, H, W)
            
        Returns:
            torch.Tensor: 图像块特征，形状为 (B, n_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        # 如果输入尺寸与期望尺寸不匹配，调整到期望尺寸
        if H != self.img_size or W != self.img_size:
            x = torch.nn.functional.interpolate(
                x, 
                size=(self.img_size, self.img_size), 
                mode='bilinear', 
                align_corners=False
            )
            H, W = self.img_size, self.img_size
        
        # 通过卷积层提取patch特征
        x = self.projection(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        
        # 重新排列维度
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    
    用于处理图像块之间的依赖关系，学习局部特征的重要性
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        """
        初始化多头自注意力层
        
        Args:
            embed_dim (int): 嵌入维度
            num_heads (int): 注意力头数
            dropout (float): Dropout概率
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 查询、键、值的线性变换
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征，形状为 (B, n_patches, embed_dim)
            
        Returns:
            torch.Tensor: 注意力输出，形状为 (B, n_patches, embed_dim)
        """
        B, N, C = x.shape
        
        # 生成Q、K、V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 (B, num_heads, N, head_dim)
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力权重
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer块 - 包含多头自注意力和前馈网络
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        """
        初始化Transformer块
        
        Args:
            embed_dim (int): 嵌入维度
            num_heads (int): 注意力头数
            mlp_ratio (float): MLP隐藏层维度比例
            dropout (float): Dropout概率
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 前馈网络
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征
            
        Returns:
            torch.Tensor: 输出特征
        """
        # 残差连接 + 多头自注意力
        x = x + self.attn(self.norm1(x))
        
        # 残差连接 + 前馈网络
        x = x + self.mlp(self.norm2(x))
        
        return x


class PatchFeatureExtractor(nn.Module):
    """
    图像块特征提取器
    
    基于Transformer架构，专门用于从打乱的图像块中提取特征
    """
    
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_channels: int = 3,
        embed_dim: int = 768, 
        num_layers: int = 12, 
        num_heads: int = 12, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.1
    ):
        """
        初始化图像块特征提取器
        
        Args:
            img_size (int): 输入图像尺寸
            patch_size (int): 图像块尺寸
            in_channels (int): 输入通道数
            embed_dim (int): 嵌入维度
            num_layers (int): Transformer层数
            num_heads (int): 注意力头数
            mlp_ratio (float): MLP隐藏层维度比例
            dropout (float): Dropout概率
        """
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化位置嵌入
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像，形状为 (B, C, H, W)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - 图像块特征序列 (B, n_patches, embed_dim)
                - 全局特征 (B, embed_dim)
        """
        B = x.shape[0]
        
        # 图像块嵌入
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # 添加位置嵌入
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # 通过Transformer层
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # 层归一化
        x = self.norm(x)
        
        # 计算全局特征（平均池化）
        global_features = x.mean(dim=1)  # (B, embed_dim)
        
        return x, global_features


class CombinedDetector(nn.Module):
    """
    综合检测器 - 结合图像块特征和全局特征
    
    这个模型能够：
    1. 接收经过图像块打乱的图像
    2. 提取局部图像块特征
    3. 结合全局特征进行最终分类
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 2,
        use_global_features: bool = True,
        use_patch_features: bool = True
    ):
        """
        初始化综合检测器
        
        Args:
            img_size (int): 输入图像尺寸
            patch_size (int): 图像块尺寸
            in_channels (int): 输入通道数
            embed_dim (int): 嵌入维度
            num_layers (int): Transformer层数
            num_heads (int): 注意力头数
            mlp_ratio (float): MLP隐藏层维度比例
            dropout (float): Dropout概率
            num_classes (int): 分类类别数
            use_global_features (bool): 是否使用全局特征
            use_patch_features (bool): 是否使用图像块特征
        """
        super().__init__()
        
        self.use_global_features = use_global_features
        self.use_patch_features = use_patch_features
        
        # 图像块特征提取器
        self.patch_extractor = PatchFeatureExtractor(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # 分类头
        if use_global_features and use_patch_features:
            # 使用全局特征 + 图像块特征
            classifier_input_dim = embed_dim + embed_dim
        elif use_global_features:
            # 只使用全局特征
            classifier_input_dim = embed_dim
        elif use_patch_features:
            # 只使用图像块特征
            classifier_input_dim = embed_dim
        else:
            raise ValueError("至少需要启用全局特征或图像块特征中的一个")
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 图像块注意力池化（用于图像块特征）
        if use_patch_features:
            self.patch_attention = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.Tanh(),
                nn.Linear(embed_dim // 4, 1),
                nn.Softmax(dim=1)
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像，形状为 (B, C, H, W)
            
        Returns:
            torch.Tensor: 分类logits，形状为 (B, num_classes)
        """
        # 提取图像块特征
        patch_features, global_features = self.patch_extractor(x)
        
        features_to_concat = []
        
        # 处理全局特征
        if self.use_global_features:
            features_to_concat.append(global_features)
        
        # 处理图像块特征
        if self.use_patch_features:
            # 使用注意力机制聚合图像块特征
            attention_weights = self.patch_attention(patch_features)  # (B, n_patches, 1)
            weighted_patch_features = (patch_features * attention_weights).sum(dim=1)  # (B, embed_dim)
            features_to_concat.append(weighted_patch_features)
        
        # 拼接特征
        if len(features_to_concat) > 1:
            combined_features = torch.cat(features_to_concat, dim=1)
        else:
            combined_features = features_to_concat[0]
        
        # 分类
        logits = self.classifier(combined_features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> dict:
        """
        获取中间特征（用于分析和可视化）
        
        Args:
            x (torch.Tensor): 输入图像
            
        Returns:
            dict: 包含各种特征的字典
        """
        with torch.no_grad():
            patch_features, global_features = self.patch_extractor(x)
            
            features = {
                'patch_features': patch_features,
                'global_features': global_features
            }
            
            if self.use_patch_features:
                attention_weights = self.patch_attention(patch_features)
                weighted_patch_features = (patch_features * attention_weights).sum(dim=1)
                features['weighted_patch_features'] = weighted_patch_features
                features['attention_weights'] = attention_weights
            
            return features


def create_combined_detector(
    img_size: int = 224,
    patch_size: int = 16,
    num_classes: int = 2,
    model_size: str = 'small'
) -> CombinedDetector:
    """
    创建综合检测器的工厂函数
    
    Args:
        img_size (int): 输入图像尺寸
        patch_size (int): 图像块尺寸
        num_classes (int): 分类类别数
        model_size (str): 模型大小 ('small', 'medium', 'large')
        
    Returns:
        CombinedDetector: 配置好的检测器模型
    """
    configs = {
        'small': {
            'embed_dim': 384,
            'num_layers': 6,
            'num_heads': 6,
            'mlp_ratio': 4.0,
            'dropout': 0.1
        },
        'medium': {
            'embed_dim': 512,
            'num_layers': 8,
            'num_heads': 8,
            'mlp_ratio': 4.0,
            'dropout': 0.1
        },
        'large': {
            'embed_dim': 768,
            'num_layers': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
            'dropout': 0.1
        }
    }
    
    config = configs[model_size]
    
    return CombinedDetector(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        **config
    )


# 测试函数
def test_models():
    """测试模型功能"""
    print("测试图像块特征提取器...")
    
    # 创建测试数据
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 224, 224)
    
    # 测试PatchFeatureExtractor
    patch_extractor = PatchFeatureExtractor(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        num_layers=6,
        num_heads=6
    )
    
    patch_features, global_features = patch_extractor(test_input)
    print(f"图像块特征形状: {patch_features.shape}")
    print(f"全局特征形状: {global_features.shape}")
    
    # 测试CombinedDetector
    print("\n测试综合检测器...")
    detector = create_combined_detector(
        img_size=224,
        patch_size=16,
        num_classes=2,
        model_size='small'
    )
    
    logits = detector(test_input)
    print(f"分类logits形状: {logits.shape}")
    
    # 测试特征提取
    features = detector.get_features(test_input)
    print(f"提取的特征键: {list(features.keys())}")


if __name__ == "__main__":
    test_models()
