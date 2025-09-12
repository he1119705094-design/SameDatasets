# AlignedForensics + FakeImageDetection 集成训练系统

## 🎯 项目概述

本项目成功将两篇重要论文的核心思想进行了结合：

1. **AlignedForensics**: 《Aligned Datasets Improve Detection of Latent Diffusion-Generated Images》
   - 核心：使用LDM自编码器重建真实图像生成高度对齐的伪造图像
   - 优势：确保训练数据中真实和伪造图像的唯一显著差异是生成模型引入的伪影

2. **FakeImageDetection**: 《打破语义伪影以进行广义AI生成的图像检测》
   - 核心：通过图像块打乱破坏图像的全局语义信息
   - 优势：迫使检测器关注局部"生成器伪影"，提高跨场景泛化能力

## 🚀 核心创新

### 1. 图像块打乱技术 (ImagePatchShuffle)
- **位置**: `data_transforms.py`
- **功能**: 将图像分割成固定大小的块，随机打乱后重新组合
- **效果**: 破坏全局语义，迫使模型关注局部特征

### 2. 基于Transformer的检测器 (CombinedDetector)
- **位置**: `networks/new_models.py`
- **架构**: 
  - PatchEmbedding: 将图像块转换为特征向量
  - MultiHeadSelfAttention: 学习图像块间的关系
  - TransformerBlock: 处理图像块序列
  - 综合分类器: 结合局部和全局特征

### 3. 自适应数据处理
- **位置**: `utils/dataset.py`
- **功能**: 自动处理不同尺寸的图像，确保训练稳定性

## 📁 文件结构

```
training_code/
├── data_transforms.py          # 图像块打乱变换
├── networks/
│   └── new_models.py          # CombinedDetector模型
├── utils/
│   ├── patch_training.py      # 基于图像块的训练模型
│   ├── dataset.py             # 数据处理和加载
│   └── processing.py          # 图像预处理（已修改）
├── train.py                   # 主训练脚本（已修改）
├── test_integration.py        # 集成测试脚本
├── run_example.py             # 使用示例
└── README_INTEGRATION.md      # 本文档
```

## 🛠️ 使用方法

### 1. 传统ResNet模型（原有方法）
```bash
python train.py --name traditional_test --dataroot ../datasets --arch res50nodown --batch_size 8 --lr 0.0001
```

### 2. ResNet + 图像块打乱
```bash
python train.py --name patch_shuffle_test --dataroot ../datasets --arch res50nodown --batch_size 8 --lr 0.0001 --patch_size 16
```

### 3. CombinedDetector模型（推荐）
```bash
python train.py --name combined_test --dataroot ../datasets --use_patch_model --model_size small --batch_size 8 --lr 0.0001 --patch_size 16
```

### 4. 最佳配置（AlignedForensics + FakeImageDetection）
```bash
python train.py --name best_test --dataroot ../datasets --use_patch_model --model_size medium --batch_size 8 --lr 0.0001 --patch_size 16 --batched_syncing --use_inversions
```

## ⚙️ 关键参数说明

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--use_patch_model` | 使用CombinedDetector模型 | False | True |
| `--model_size` | 模型大小 | small | small/medium/large |
| `--patch_size` | 图像块大小 | 16 | 16 |
| `--batched_syncing` | 启用批次同步 | False | True |
| `--use_inversions` | 使用LDM重建图像 | False | True |
| `--batch_size` | 批次大小 | 64 | 8-16 |
| `--lr` | 学习率 | 0.0001 | 0.0001 |

## 🧪 测试验证

### 运行集成测试
```bash
python test_integration.py
```

### 查看使用示例
```bash
python run_example.py
```

## 📊 性能特点

### CombinedDetector模型参数
- **Small**: ~11M 参数
- **Medium**: ~15M 参数  
- **Large**: ~25M 参数

### 训练效果
- ✅ 支持不同尺寸图像输入
- ✅ 自动图像块打乱
- ✅ Transformer架构处理图像块序列
- ✅ 结合局部和全局特征分类
- ✅ 兼容AlignedForensics的数据对齐策略

## 🔧 技术实现细节

### 1. 图像块打乱算法
```python
# 将图像分割成patches
patches = tensor.view(C, num_patches_h, patch_size, num_patches_w, patch_size)
# 随机打乱
shuffle_indices = torch.randperm(num_patches)
shuffled_patches = patches[:, shuffle_indices, :, :]
# 重新组合
shuffled_tensor = shuffled_patches.view(C, H, W)
```

### 2. Transformer架构
```python
# 图像块嵌入
x = self.patch_embed(x)  # (B, n_patches, embed_dim)
# 添加位置嵌入
x = x + self.pos_embed
# Transformer处理
for transformer_block in self.transformer_blocks:
    x = transformer_block(x)
# 全局特征聚合
global_features = x.mean(dim=1)
```

### 3. 自适应尺寸处理
```python
# 自动调整到期望尺寸
if H != self.img_size or W != self.img_size:
    x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear')
```

## 🎉 成功案例

所有测试均已通过：
- ✅ 图像块打乱功能测试
- ✅ CombinedDetector模型测试  
- ✅ PatchTrainingModel测试
- ✅ 数据处理集成测试
- ✅ 完整训练流程测试

## 📈 预期效果

结合两篇论文的核心思想，预期能够：

1. **提高泛化能力**: 通过图像块打乱破坏全局语义，避免过拟合
2. **增强检测精度**: 使用Transformer架构更好地学习局部特征
3. **改善数据质量**: 利用AlignedForensics的数据对齐策略
4. **提升鲁棒性**: 结合局部和全局特征进行综合判断

## 🚀 下一步建议

1. **超参数调优**: 尝试不同的patch_size和model_size
2. **数据增强**: 结合更多数据增强策略
3. **模型融合**: 尝试ensemble多个模型
4. **跨域测试**: 在不同数据集上验证泛化能力

---

**注意**: 这是一个研究项目，建议在正式使用前进行充分的实验验证。
