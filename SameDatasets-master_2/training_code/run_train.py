import os
import subprocess

# ===== 配置参数 =====
os.environ["DATASET"] = r'D:\Model\AlignedForensics-master_1\aligned_dataset'  # 替换为实际数据集路径
os.environ["HBB_sync"] = r'D:\Model\AlignedForensics-master\checkpoint'  # 替换为预训练检测器路径
os.environ["EXP_NAME"] = "Second_test"  # 自定义实验名称
os.environ["LDM_DS_NAME"] = "checkpoints_dir"  # 检查点目录


# ===== 训练命令配置 =====
# 对齐数据集进行的命令
def run_training_command(device_id=0, sync_mode=False, full_dataset=False):
    cmd = [
        "python", "train.py",
        f"--name={os.environ['EXP_NAME'] if not full_dataset else os.environ['LDM_DS_NAME']}",
        "--arch", "res50nodown",
        "--cropSize", "96",
        "--norm_type", "resnet",
        "--resize_size", "256",
        "--resize_ratio", "0.75",
        "--blur_sig", "0.0,3.0",
        "--cmp_method", "cv2,pil",
        "--cmp_qual", "30,100",
        "--resize_prob", "0.2",
        "--jitter_prob", "0.8",
        "--colordist_prob", "0.2",
        "--cutout_prob", "0.2",
        "--noise_prob", "0.2",
        "--blur_prob", "0.5",
        "--cmp_prob", "0.5",
        "--rot90_prob", "1.0",
        f"--dataroot={os.environ['DATASET']}",
        "--batch_size", "8" ,
        "--earlystop_epoch", "5" if not sync_mode else "10",
        "--seed", "14" if not sync_mode else "17",
        *(["--stay_positive", "clamp"] if not sync_mode else []),
        *([f"--ckpt={os.environ['HBB_sync']}"] if not sync_mode else []),
        *(["--fix_backbone"] if not sync_mode else []),
        *(["--final_dropout", "0.0"] if not sync_mode else []),
        "--use_inversions",
        *(["--batched_syncing", "--data_cap=1000"] if sync_mode else [])
    ]

    # 设置CUDA设备
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)

    # 执行命令
    print("\n执行命令:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)

# 不对齐数据集时候进行的训练
def run_custom_command():
    cmd = [
        "python", "train.py",
        f"--name={os.environ['LDM_DS_NAME']}",
        "--arch", "res50nodown",
        "--cropSize", "96",
        "--norm_type", "resnet",
        "--resize_size", "256",
        "--resize_ratio", "0.75",
        "--blur_sig", "0.0,3.0",
        "--cmp_method", "cv2,pil",
        "--cmp_qual", "30,100",
        "--resize_prob", "0.2",
        "--jitter_prob", "0.8",
        "--colordist_prob", "0.2",
        "--cutout_prob", "0.2",
        "--noise_prob", "0.2",
        "--blur_prob", "0.5",
        "--cmp_prob", "0.5",
        "--rot90_prob", "1.0",
        f"--dataroot={os.environ['DATASET']}",
        #"--batch_size", "128",
        "--batch_size", "16",
        "--earlystop_epoch", "10",
        "--use_inversions",
        "--seed", "17"
    ]

    # 设置 CUDA 设备
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # 执行命令
    print("\n执行命令:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


# ===== 选择要运行的训练模式 =====
if __name__ == "__main__":
    # 模式选择 (取消注释要运行的命令)

    # 1. 同步模式训练（1000真实+1000生成图像）
    #run_training_command(device_id=0, sync_mode=True)

    # 2. 全数据集非同步训练
     #run_training_command(device_id=0, full_dataset=True)
     run_custom_command()
    # 3. ICML实验（Stay-Positive方法）
    #run_training_command(device_id=0)  # 默认运行ICML实验