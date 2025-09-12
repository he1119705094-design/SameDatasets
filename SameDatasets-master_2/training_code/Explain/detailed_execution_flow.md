# 详细执行流程分析

## 🚀 您的命令执行流程

```bash
python train.py --name traditional_test --dataroot ../datasets --arch res50nodown --batch_size 8 --lr 0.0001
```

## 📋 逐步执行分析

### 1. 启动阶段
```
train.py 启动
    ↓
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置使用GPU 0
    ↓
解析命令行参数
```

### 2. 参数解析结果
```python
opt.name = "traditional_test"
opt.dataroot = "../datasets"
opt.arch = "res50nodown"
opt.batch_size = 8
opt.lr = 0.0001
opt.patch_size = 16  # 默认值
opt.use_patch_model = False  # 默认值
opt.model_size = "small"  # 默认值
```

### 3. 环境设置
```python
torch.manual_seed(opt.seed)  # 设置随机种子
```

### 4. 数据加载
```
create_dataloader(opt, subdir="train", is_train=True)
    ↓
dataroot = "../datasets/train"
    ↓
get_dataset(opt, dataroot)  # 因为没指定batched_syncing
    ↓
make_processing(opt)  # 创建数据变换
```

### 5. 数据处理流程
```
原始图像
    ↓
make_pre(opt)  # 预处理：缩放
    ↓
make_aug(opt)  # 数据增强：旋转、翻转等
    ↓
make_post(opt)  # 后处理：裁剪、调整尺寸
    ↓
make_normalize(opt)  # 归一化：ToTensor + Normalize
    ↓
ImagePatchShuffle(patch_size=16)  # 图像块打乱
    ↓
最终tensor (B, C, H, W)
```

### 6. 模型创建
```
opt.use_patch_model = False
    ↓
执行: print("使用传统ResNet训练模型")
    ↓
model = TrainingModel(opt, subdir="traditional_test")
```

### 7. TrainingModel初始化
```python
self.device = torch.device('cuda:0')  # 使用GPU
self.model = create_architecture("res50nodown", ...)  # 创建ResNet-50
self.loss_fn = torch.nn.BCEWithLogitsLoss()  # 二分类损失
self.optimizer = torch.optim.Adam(...)  # Adam优化器
```

### 8. 训练循环
```
for epoch in range(start_epoch, 1001):  # 默认1000个epoch
    ↓
    if epoch > start_epoch:
        # 训练阶段
        for data in train_data_loader:
            loss = model.train_on_batch(data)
            # 显示训练损失
        # 保存模型
    ↓
    # 验证阶段
    y_true, y_pred, y_path = model.predict(valid_data_loader)
    acc = balanced_accuracy_score(y_true, y_pred > 0.0)
    # 记录验证准确率
    ↓
    # 早停检查
    if early_stopping(acc):
        # 保存最佳模型
        if early_stopping.early_stop:
            # 降低学习率或停止训练
```

### 9. train_on_batch详细流程
```python
def train_on_batch(self, data):
    self.total_steps += 1
    self.model.train()  # 设置为训练模式
    
    # 数据准备
    input = data['img'].to(self.device)  # 移动到GPU
    label = data['target'].to(self.device).float()
    
    # 前向传播
    output, feats = self.model(input, return_feats=False)
    
    # 损失计算
    loss = self.loss_fn(output.squeeze(1), label)
    
    # 反向传播
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    return loss.cpu()
```

### 10. 验证流程
```python
def predict(self, data_loader):
    model = self.model.eval()  # 设置为评估模式
    with torch.no_grad():
        for data in data_loader:
            img = data['img']
            label = data['target'].cpu().numpy()
            
            # 前向传播
            out_tens, _ = model(img.to(self.device))
            out_tens = out_tens.cpu().numpy()[:, -1]
            
            # 收集预测结果
            y_pred.extend(out_tens.tolist())
            y_true.extend(label.tolist())
    
    return y_true, y_pred, y_path
```

## 🔍 关键执行点

### 图像块打乱的应用
- 在数据预处理阶段，每个图像都会经过 `ImagePatchShuffle(patch_size=16)`
- 图像被分割成16×16的小块，随机打乱后重新组合
- 这破坏了图像的全局语义，迫使模型关注局部特征

### ResNet模型处理
- 使用 `res50nodown` 架构（ResNet-50 without downsampling）
- 模型输出形状: `(batch_size, 1)`
- 使用 `BCEWithLogitsLoss` 进行二分类

### 设备使用
- 默认使用GPU 0进行训练
- 所有tensor都会移动到GPU: `.to(self.device)`

## 📊 输出信息解读

当您运行命令时，会看到类似这样的输出：
```
normalize RESNET
添加图像块打乱变换，patch_size=16
CLASSES: ['latent_diffusion_noise2image_churches', 'latent_diffusion_noise2image_FFHQ', 'latent_diffusion_text2img_set2']
#images     20 in ../datasets\valid/latent_diffusion_noise2image_churches
#images     22 in ../datasets\valid/latent_diffusion_noise2image_FFHQ
normalize RESNET
添加图像块打乱变换，patch_size=16
CLASSES: ['.']
#images    400 in ../datasets\train/.
RandomSampler: # [1 1]

# validation batches = 6
#   training batches = 50
使用传统ResNet训练模型
Arch: res50nodown with #trainable 23511397
lr: 0.0001
Validation ...
After 0 epoches: val acc = 0.475
Train loss: 0.6903: 100%|███████████████████████████████████████████████████████████████████████████████| 100/100 [01:06<00:00,  1.50it/s] 
Validation ...
After 1 epoches: val acc = 0.3704545454545455
```

这表示：
- 使用了ResNet模型（不是CombinedDetector）
- 添加了图像块打乱变换
- 训练集有400张图像，验证集有42张图像
- 模型有约2350万个可训练参数
- 学习率为0.0001
- 第一个epoch后验证准确率为37%
