import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from torchvision.transforms import v2
from torchvision.transforms import functional as TF
from torchvision import tv_tensors

from .model_factory import CLASS_COLORS


COLOR_TO_CLASS = {
    color: idx for idx, color in enumerate(CLASS_COLORS)
}


def rgb_mask_to_class_indices(mask_rgb):
    class_mask = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
    unknown = np.ones(mask_rgb.shape[:2], dtype=bool)

    for color, class_id in COLOR_TO_CLASS.items():
        matches = np.all(mask_rgb == color, axis=-1)
        class_mask[matches] = class_id
        unknown &= ~matches

    if unknown.any():
        unknown_colors = np.unique(mask_rgb[unknown].reshape(-1, 3), axis=0)[:5]
        raise ValueError(f"Mask contains unknown RGB label colors: {unknown_colors.tolist()}")

    return class_mask

class DataGen(Dataset):
    def __init__(self, img_path, mask_path, X, mean, std, transform=None, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.mean = mean
        self.std = std
        self.transform = transform
        self.patches = patch

        # Predefine final normalization to avoid doing it every __getitem__
        self.normalize = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.mean, std=self.std)
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img_file = os.path.join(self.img_path, self.X[idx])
        mask_file = os.path.join(self.mask_path, self.X[idx])

        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_file}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_file}")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = rgb_mask_to_class_indices(mask)

        # Convert to pure PyTorch v2 tv_tensors (CHW format)
        img_tensor = tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1))
        mask_tensor = tv_tensors.Mask(torch.from_numpy(mask))

        if self.transform is not None:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        # Convert image to float32 and normalize
        img_tensor = self.normalize(img_tensor)

        # Convert mask to long
        mask_tensor = mask_tensor.long()

        if self.patches:
            img_tensor, mask_tensor = self.get_img_patches(img_tensor, mask_tensor)

        return img_tensor, mask_tensor

    def get_img_patches(self, img, mask):
        kh, kw = 512, 768  # Kernel size
        dh, dw = 512, 768  # Strides

        img_patches = img.unfold(1, kh, dh).unfold(2, kw, dw)
        img_patches = img_patches.contiguous().view(3, -1, kh, kw)
        img_patches = img_patches.permute(1, 0, 2, 3)

        mask_patches = mask.unfold(0, kh, dh).unfold(1, kw, dw)
        mask_patches = mask_patches.contiguous().view(-1, kh, kw)

        return img_patches, mask_patches


class TestDataGen(Dataset):
    def __init__(self, img_path, mask_path, X, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img_file = os.path.join(self.img_path, self.X[idx])
        mask_file = os.path.join(self.mask_path, self.X[idx])

        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_file}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_file}")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = rgb_mask_to_class_indices(mask)

        # Convert to pure PyTorch v2 tv_tensors
        img_tensor = tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1))
        mask_tensor = tv_tensors.Mask(torch.from_numpy(mask))

        if self.transform is not None:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        # Return PIL Image since utils.py evaluates TestDataset expecting PIL 
        img_pil = TF.to_pil_image(img_tensor)
        mask_tensor = mask_tensor.long()

        return img_pil, mask_tensor
