# 代码执行逻辑详解

## 🚀 完整执行流程

```
1. 启动 train.py
   ↓
2. 解析命令行参数
   ├── --use_patch_model (是否使用新模型)
   ├── --patch_size (图像块大小)
   ├── --model_size (模型大小)
   └── 其他参数...
   ↓
3. 设置随机种子和设备
   ↓
4. 创建数据加载器
   ├── 训练集: create_dataloader(opt, subdir="train", is_train=True)
   └── 验证集: create_dataloader(opt, subdir="valid", is_train=False)
   ↓
5. 选择训练模型类
   ├── 如果 --use_patch_model: PatchTrainingModel
   └── 否则: TrainingModel
   ↓
6. 开始训练循环
   ├── 训练阶段 (train_on_batch)
   ├── 验证阶段 (predict)
   └── 早停检查
```

## 📊 数据处理流程

```
原始图像
   ↓
预处理 (resize, crop等)
   ↓
数据增强 (旋转、翻转、颜色变换等)
   ↓
归一化 (ToTensor + Normalize)
   ↓
图像块打乱 (ImagePatchShuffle) ← 新增功能
   ↓
最终tensor (B, C, H, W)
```

## 🏗️ 模型架构对比

### 传统ResNet模型
```
输入图像 → ResNet骨干网络 → 全局平均池化 → 全连接层 → 输出
```

### CombinedDetector模型 (新增)
```
输入图像 → 尺寸调整 → PatchEmbedding → Transformer层 → 特征聚合 → 分类器 → 输出
```

## 🔧 关键组件说明

### 1. ImagePatchShuffle (图像块打乱)
- **位置**: `data_transforms.py`
- **作用**: 将图像分割成小块，随机打乱后重新组合
- **目的**: 破坏全局语义，迫使模型关注局部特征

### 2. CombinedDetector (综合检测器)
- **位置**: `networks/new_models.py`
- **架构**: 基于Transformer的图像块处理模型
- **特点**: 能够处理不同尺寸的图像，结合局部和全局特征

### 3. PatchTrainingModel (训练模型)
- **位置**: `utils/patch_training.py`
- **作用**: 封装CombinedDetector的训练逻辑
- **功能**: 支持前向传播、损失计算、反向传播、模型保存等

## 🎯 核心创新点

1. **图像块打乱**: 破坏全局语义，提高泛化能力
2. **Transformer架构**: 更好地学习图像块间的关系
3. **自适应处理**: 自动处理不同尺寸的图像
4. **模块化设计**: 可以独立使用各个组件

## 📝 使用示例

### 使用传统ResNet
```bash
python train.py --name resnet_test --dataroot ../datasets --arch res50nodown --batch_size 8 --lr 0.0001
```

### 使用CombinedDetector
```bash
python train.py --name patch_test --dataroot ../datasets --use_patch_model --model_size small --batch_size 8 --lr 0.0001 --patch_size 16
```

### 使用最佳配置
```bash
python train.py --name best_test --dataroot ../datasets --use_patch_model --model_size medium --batch_size 8 --lr 0.0001 --patch_size 16 --batched_syncing --use_inversions
```
