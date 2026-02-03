# =================================================================================================
#
# A Unified and Robust PyTorch Dataset Class for AI-Generated Image Detection
#
# Final Version: Yuncheng Guo
#
# =================================================================================================

import io
import os
import json
import random
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np

import warnings

from tqdm import tqdm

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

class GenerativeImageDataset(Dataset):
    """A unified and robust PyTorch Dataset class for detecting AI-generated images."""
    def __init__(self, root: str, is_train: bool,
                 label: int = None, category2label: int = None,
                 resolution: int = 336, model_name: str = None,
                 real_folder_name: str = '0_real', fake_folder_name: str = '1_fake'):

        self.root = root
        self.is_train = is_train
        self.resolution = resolution
        self.mean = [0.485, 0.456, 0.406] if "DINO" in model_name else [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.229, 0.224, 0.225] if "DINO" in model_name else [0.26862954, 0.26130258, 0.27577711]
        self.data_list = []
        self.explicit_label = label
        self.category2label = category2label

        self.real_folder = real_folder_name
        self.fake_folder = fake_folder_name

        self._init_transforms()
        self._load_data()

        if not self.data_list:
            raise RuntimeError(f"No data found for the specified parameters at root: {self.root}")


    def _init_transforms(self):
        self.train_transform = transforms.Compose([ 
            transforms.Resize([512, 512]),
            transforms.Resize([self.resolution, self.resolution]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.Resize([self.resolution, self.resolution]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])


    def _load_data(self):
        print(f"Recursively scanning for data in: {self.root}")
        print(f" - Real folder: '{self.real_folder}', Fake folder: '{self.fake_folder}'")

        # os.walk is the perfect tool for recursive traversal.
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames.sort()
            filenames.sort()

            relative_path = os.path.relpath(dirpath, self.root)
            path_parts = relative_path.split(os.sep)

            label_for_this_dir = None

            if self.explicit_label is not None:
                # If an override label is given, we load from any folder.
                label_for_this_dir = self.explicit_label

            elif self.real_folder and self.real_folder in path_parts:
                label_for_this_dir = 0
            elif self.fake_folder and self.fake_folder in path_parts:
                label_for_this_dir = 1

            # If a target folder (real or fake) is found, process all images within it.
            if label_for_this_dir is not None:

                category = path_parts[0] if path_parts and path_parts[0] != '.' else 'unknown'

                for img_name in filenames:
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        full_path = os.path.join(dirpath, img_name)
                        self.data_list.append({
                            "image_path": full_path,
                            "label": label_for_this_dir,
                            "category": category
                        })

        if self.data_list:
            real_count = sum(1 for item in self.data_list if item['label'] == 0)
            fake_count = sum(1 for item in self.data_list if item['label'] == 1)
            total_count = len(self.data_list)
            
            print(f"Found {total_count} images:")
            print(f" - Real images: {real_count}")
            print(f" - Fake images: {fake_count}")
            if total_count > 0:
                print(f" - Real/Fake ratio: {real_count/total_count:.2%}/{fake_count/total_count:.2%}")
        else:
            print("Warning: No images found. Check your root path and folder names.")


    def _load_rgb(self, file_path: str) -> Image.Image:
        try:
            pil_img = Image.open(file_path).convert('RGB')
            return pil_img
        except Exception as e:
            raise IOError(f"Failed to load image '{file_path}' PIL.") from e


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        sample = self.data_list[index]
        image_path = sample['image_path']
        target = sample['label']

        category_label = -1
    
        if 'category' in sample:
            if self.category2label is not None and sample['category'] in self.category2label:
                category_label = self.category2label[sample['category']]
            
        try:
            image = self._load_rgb(image_path)
            if self.is_train:
                image_tensor = self.train_transform(image)
            else:
                image_tensor = self.eval_transform(image)
            
            return image_tensor, torch.tensor(int(target), dtype=torch.long), image_path, torch.tensor(int(category_label), dtype=torch.long)

        except Exception as e:
            warnings.warn(f"Error loading image '{image_path}': {e}. Skipping and loading a random sample.")
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))
