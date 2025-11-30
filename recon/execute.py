from PIL import Image, ImageFilter
import numpy as np
from recon import *
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


def save_images(images_tensor, input_path, names, original_path_prefix=None, new_path_prefix=None):
    output_folder = input_path.replace(original_path_prefix, new_path_prefix)

    os.makedirs(output_folder, exist_ok=True)
    for idx, image_tensor in enumerate(images_tensor):
        name = names[idx].split('/')[-1]
        name = name.replace('.jpg', '.png')
        if torch.is_tensor(image_tensor):
            image_pil = transforms.ToPILImage()(image_tensor)
        else:
            image_pil = image_tensor
        image_path = os.path.join(output_folder, name)
        image_pil.save(image_path)


def create_dataloader(image_folder, batch_size=32, shuffle=True, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    dataset = UnlabeledImageDataset(image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def recon(pipe, dataloader, ae, seed, args, tools):
    for batch_idx, (images, names) in tqdm(enumerate(dataloader), total=len(dataloader), desc='Reconstructing images'):
        if batch_idx == 0:
            print(f" - Images shape: {images.shape}")
        with torch.no_grad():
            recons = reconstruct_simple(x=images.to(device), ae=ae, seed=seed, steps=args.steps, tools=tools)
        save_images(images_tensor=images, input_path=args.input_folder, names=names, 
                    original_path_prefix='unlabeled2017', 
                    new_path_prefix=f'0_real_CoCo_{images.shape[-1]}x{images.shape[-1]}/0_real_{args.repo_id}')
        save_images(images_tensor=recons, input_path=args.input_folder, names=names, 
                    original_path_prefix='unlabeled2017', 
                    new_path_prefix=f'1_fake_CoCo_{images.shape[-1]}x{images.shape[-1]}/1_fake_{args.repo_id}')


def main():
    parser = argparse.ArgumentParser(description='Create a DataLoader from an image folder dataset.')
    parser.add_argument('--input_folder', type=str, help='Path to the input folder containing images.')
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



