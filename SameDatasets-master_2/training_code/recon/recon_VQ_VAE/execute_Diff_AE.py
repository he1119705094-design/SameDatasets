from PIL import Image, ImageFilter
import numpy as np
from recon_Diff_AE import *
from tqdm import tqdm
import torch
import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import imageio


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UnlabeledImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder)
                            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

def get_vae(repo_id):
    print(f"Loading VAE from {repo_id}...")
    vae = AutoencoderKL.from_pretrained(
        repo_id,
        subfolder="vae",  # 注意这里是"vae"不是"vqvae"
        cache_dir="weights",
        use_safetensors=False,
        local_files_only=False
    )
    print("✅ VAE 加载成功！")
    print(vae)
    return vae

def save_images(images_tensor, output_folder, names):
    os.makedirs(output_folder, exist_ok=True)
    for idx, image_tensor in enumerate(images_tensor):
        name = os.path.basename(names[idx])
        name = name.replace('.jpg', '.png')  # 或根据实际情况处理
        if torch.is_tensor(image_tensor):
            image_pil = transforms.ToPILImage()(image_tensor)
        else:
            image_pil = image_tensor
        image_path = os.path.join(output_folder, name)
        image_pil.save(image_path)






def create_dataloader(image_folder, batch_size=32, shuffle=True, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = UnlabeledImageDataset(image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


# def recon(pipe, dataloader, ae, seed, args, tools):
#     for batch_idx, (images, names) in enumerate(dataloader):
#         if batch_idx == 0:
#             print(f"Batch {batch_idx + 1}:")
#             print(f" - Images shape: {images.shape}")
#         with torch.no_grad():
#             recons = reconstruct_simple(x=images.to(device), ae=ae, seed=seed, steps=args.steps, tools=tools)
#         save_images(images_tensor=recons, input_path=args.input_folder, names=names)

def recon(pipe, dataloader, ae, seed, args, tools):
    for batch_idx, (images, names) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f" - Images shape: {images.shape}")
        with torch.no_grad():
            recons = reconstruct_simple(x=images.to(device), ae=ae, seed=seed, steps=args.steps, tools=tools)
        save_images(images_tensor=recons, output_folder=args.output_folder, names=names)


# def main():
#     parser = argparse.ArgumentParser(description='Create a DataLoader from an image folder dataset.')
#     parser.add_argument('--input_folder', type=str, help='Path to the input folder containing images.')
#     parser.add_argument('--repo_id', type=str, help='Correct stable diffusion autoencoder')
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader.')
#     parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the data in the DataLoader.')
#     parser.add_argument('--steps', type=int, default=None, help='Number of backward steps in DDIM inversion')
#     parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader.')
#     args = parser.parse_args()
#     seed = 42
#     dataloader = create_dataloader(args.input_folder, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
#
#     ae = get_vae(repo_id=args.repo_id).to(device)
#     tools = None
#     recon(pipe=None, dataloader=dataloader, ae=ae, seed=seed, args=args, tools=tools)

def main():
    parser = argparse.ArgumentParser(description='Create a DataLoader from an image folder dataset.')
    parser.add_argument('--input_folder', type=str, help='Path to the input folder containing images.')
    parser.add_argument('--output_folder', type=str, help='Path to save the reconstructed images.')
    parser.add_argument('--repo_id', type=str, help='Correct stable diffusion autoencoder')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader.')
    parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the data in the DataLoader.')
    parser.add_argument('--steps', type=int, default=None, help='Number of backward steps in DDIM inversion')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader.')
    args = parser.parse_args()

    seed = 42
    dataloader = create_dataloader(args.input_folder, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
    ae = get_vae(repo_id=args.repo_id).to(device)
    tools = None
    recon(pipe=None, dataloader=dataloader, ae=ae, seed=seed, args=args, tools=tools)


if __name__ == "__main__":
    main()