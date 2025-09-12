import os

path = r"D:\Model\AlignedForensics-master\datasets\valid"
dirs = os.listdir( path )
print(dirs)
print("---------------------------------")
print('CLASSES:', dirs)
# 输出所有文件和文件夹
for file in dirs:
   print (file)
print("----------------------------------")
import torch
loss_fn = torch.nn.BCEWithLogitsLoss()
logits = torch.tensor([1.5]*10)  # 模型输出都为1.5
print(logits)
labels = torch.ones(10)           # 真实标签全是1
print(labels)
loss = loss_fn(logits, labels)
print(loss)  # 输出损失值，数字越小越好