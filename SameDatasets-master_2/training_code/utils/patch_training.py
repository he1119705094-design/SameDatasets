"""
基于图像块的训练模型类

这个模块扩展了原有的TrainingModel类，支持新的CombinedDetector模型
"""

import os
import torch
import torch.nn as nn
import numpy as np
import tqdm
from .losses import SupConLoss
from networks import create_architecture, count_parameters
from networks.new_models import CombinedDetector, create_combined_detector
import matplotlib.pyplot as plt
import pprint
import random
import torch.nn.init as init


class PatchTrainingModel(torch.nn.Module):
    """
    基于图像块的训练模型类
    
    支持使用CombinedDetector模型进行训练
    """

    def __init__(self, opt, subdir='.'):
        super(PatchTrainingModel, self).__init__()

        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, subdir)
        self.device = torch.device('cpu') if opt.no_cuda else torch.device('cuda:0')
        
        # 根据参数选择模型架构
        if opt.use_patch_model:
            print(f"使用基于图像块的CombinedDetector模型，模型大小: {opt.model_size}")
            # 创建CombinedDetector模型
            self.model = create_combined_detector(
                img_size=opt.resize_size if hasattr(opt, 'resize_size') else 224,
                patch_size=opt.patch_size,
                num_classes=1,  # 二分类任务
                model_size=opt.model_size
            )
        else:
            # 使用原有的ResNet架构
            self.model = create_architecture(
                opt.arch, 
                pretrained=not opt.start_fresh,  
                num_classes=1,
                leaky=opt.use_leaky,
                ckpt=opt.ckpt, 
                use_proj=self.opt.use_proj, 
                proj_ratio=opt.proj_ratio,
                dropout=opt.final_dropout
            )
        
        num_parameters = count_parameters(self.model)
        print(f"模型架构: {opt.arch if not opt.use_patch_model else 'CombinedDetector'}，可训练参数数量: {num_parameters}")

        print('学习率:', opt.lr)
        # 二元交叉熵损失
        self.loss_fn = torch.nn.BCEWithLogitsLoss().to(self.device)

        if self.opt.fix_backbone:
            self.freeze_backbone(unfreeze_last_k=self.opt.unfreeze_last_k)
        
        # 创建优化器
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if opt.optim == "adam":
            self.optimizer = torch.optim.Adam(
                parameters, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                parameters, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("优化器应该是 [adam, sgd]")
        
        # 加载预训练模型或继续训练
        if opt.pretrain:
            self.model.load_state_dict(
                torch.load(opt.pretrain, map_location="cpu")["model"]
            )
            print("加载预训练模型: ", opt.pretrain)
        if opt.continue_epoch is not None:
            self.load_networks(opt.continue_epoch)
        
        self.model.to(self.device)

    def freeze_backbone(self, unfreeze_last_k: int = 0):
        """
        冻结骨干网络（对于CombinedDetector模型，冻结Transformer层）
        """
        if self.opt.use_patch_model:
            # 对于CombinedDetector，冻结patch_extractor的transformer_blocks
            for i, block in enumerate(self.model.patch_extractor.transformer_blocks):
                if i < len(self.model.patch_extractor.transformer_blocks) - unfreeze_last_k:
                    for param in block.parameters():
                        param.requires_grad = False
                    block.eval()
            print(f"冻结了 {len(self.model.patch_extractor.transformer_blocks) - unfreeze_last_k} 个Transformer块")
        else:
            # 原有的ResNet冻结逻辑
            backbone_blocks = [name for name, _ in self.model.named_children() if name.startswith("layer")]
            blocks_to_unfreeze = backbone_blocks[-unfreeze_last_k:] if unfreeze_last_k > 0 else []
            for name, param in self.model.named_parameters():
                block_name = name.split('.')[0]
                if block_name in blocks_to_unfreeze or 'fc' in name:
                    param.requires_grad = True
                    if 'fc' in name:
                        param.data.zero_()
                else:  
                    param.requires_grad = False
                    module = dict(self.model.named_modules())[block_name]
                    module.eval()

    def adjust_learning_rate(self, min_lr=1e-6):
        """调整学习率"""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True

    def get_learning_rate(self):
        """获取当前学习率"""
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def train_on_batch(self, data):
        """训练一个批次"""
        self.total_steps += 1
        self.model.train()

        if self.opt.batched_syncing:
            rdata = data[0]
            fdata = data[1]
            input = torch.cat((rdata['img'], fdata['img']), dim=0).to(self.device)
            label = torch.cat((rdata['target'], fdata['target']), dim=0).to(self.device).float()
        else:
            input = data['img'].to(self.device)
            label = data['target'].to(self.device).float()
        
        # 根据模型类型进行前向传播
        if self.opt.use_patch_model:
            # CombinedDetector模型
            output = self.model(input)
            # CombinedDetector输出形状是 (batch_size, 1)，需要squeeze
            if len(output.shape) > 1 and output.shape[1] == 1:
                output = output.squeeze(1)
        else:
            # 原有ResNet模型
            output, feats = self.model(input, return_feats=self.opt.use_contrastive)

        # 计算损失
        if len(output.shape) == 4:
            ss = output.shape
            loss = self.loss_fn(
                output,
                label[:, None, None, None].repeat(
                (1, int(ss[1]), int(ss[2]), int(ss[3]))
                ),
            )
        else:
            # 确保output和label的形状匹配
            if len(output.shape) > 1:
                output = output.squeeze()
            loss = self.loss_fn(output, label)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Stay-Positive Update (ICML)
        if self.opt.stay_positive == 'clamp':
            with torch.no_grad():
                if hasattr(self.model, 'classifier'):
                    # 对于CombinedDetector
                    for layer in self.model.classifier:
                        if isinstance(layer, nn.Linear):
                            layer.weight.data.clamp_(min=0)
                elif hasattr(self.model, 'fc'):
                    # 对于ResNet
                    self.model.fc.weight.data.clamp_(min=0)
        
        return loss.cpu()

    def save_networks(self, epoch):
        """保存模型"""
        save_filename = 'model_epoch_%s.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)

        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }

        torch.save(state_dict, save_path)

    def load_networks(self, epoch):
        """加载模型"""
        load_filename = 'model_epoch_%s.pth' % epoch
        load_path = os.path.join(self.save_dir, load_filename)

        print('从 %s 加载模型' % load_path)
        state_dict = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(state_dict['model'])
        self.model.to(self.device)

        try:
            self.total_steps = state_dict['total_steps']
        except:
            self.total_steps = 0

        try:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        except:
            pass

    def predict(self, data_loader):
        """预测验证集"""
        model = self.model.eval()
        with torch.no_grad():
            y_true, y_pred, y_path = [], [], []
            for data in tqdm.tqdm(data_loader):
                img = data['img']
                label = data['target'].cpu().numpy()
                paths = list(data['path'])
                
                if self.opt.use_patch_model:
                    # CombinedDetector模型
                    out_tens = model(img.to(self.device))
                    if len(out_tens.shape) > 1:
                        out_tens = out_tens.squeeze(1)
                else:
                    # 原有ResNet模型
                    out_tens, _ = model(img.to(self.device))
                
                out_tens = out_tens.cpu().numpy()
                assert label.shape == out_tens.shape

                y_pred.extend(out_tens.tolist())
                y_true.extend(label.tolist())
                y_path.extend(paths)

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return y_true, y_pred, y_path
