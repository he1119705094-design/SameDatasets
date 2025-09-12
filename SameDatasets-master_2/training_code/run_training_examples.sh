#!/bin/bash
# 示例训练脚本

echo "开始训练..."

# 示例1: 传统ResNet
echo "训练传统ResNet模型..."
python train.py \
    --name traditional_test \
    --dataroot ../datasets \
    --arch res50nodown \
    --batch_size 8 \
    --lr 0.0001 \
    --num_epoches 100

# 示例2: ResNet + 图像块打乱
echo "训练ResNet + 图像块打乱模型..."
python train.py \
    --name patch_shuffle_test \
    --dataroot ../datasets \
    --arch res50nodown \
    --batch_size 8 \
    --lr 0.0001 \
    --patch_size 16 \
    --num_epoches 100

# 示例3: CombinedDetector (推荐)
echo "训练CombinedDetector模型..."
python train.py \
    --name combined_detector_test \
    --dataroot ../datasets \
    --use_patch_model \
    --model_size small \
    --batch_size 8 \
    --lr 0.0001 \
    --patch_size 16 \
    --num_epoches 100

# 示例4: 最佳配置
echo "训练最佳配置模型..."
python train.py \
    --name best_test \
    --dataroot ../datasets \
    --use_patch_model \
    --model_size medium \
    --batch_size 8 \
    --lr 0.0001 \
    --patch_size 16 \
    --batched_syncing \
    --use_inversions \
    --num_epoches 100

echo "训练完成！"
