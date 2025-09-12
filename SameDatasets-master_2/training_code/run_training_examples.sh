#!/bin/bash
# ʾ��ѵ���ű�

echo "��ʼѵ��..."

# ʾ��1: ��ͳResNet
echo "ѵ����ͳResNetģ��..."
python train.py \
    --name traditional_test \
    --dataroot ../datasets \
    --arch res50nodown \
    --batch_size 8 \
    --lr 0.0001 \
    --num_epoches 100

# ʾ��2: ResNet + ͼ������
echo "ѵ��ResNet + ͼ������ģ��..."
python train.py \
    --name patch_shuffle_test \
    --dataroot ../datasets \
    --arch res50nodown \
    --batch_size 8 \
    --lr 0.0001 \
    --patch_size 16 \
    --num_epoches 100

# ʾ��3: CombinedDetector (�Ƽ�)
echo "ѵ��CombinedDetectorģ��..."
python train.py \
    --name combined_detector_test \
    --dataroot ../datasets \
    --use_patch_model \
    --model_size small \
    --batch_size 8 \
    --lr 0.0001 \
    --patch_size 16 \
    --num_epoches 100

# ʾ��4: �������
echo "ѵ���������ģ��..."
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

echo "ѵ����ɣ�"
