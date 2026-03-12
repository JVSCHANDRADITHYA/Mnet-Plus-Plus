import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class DehazeDataset(Dataset):

    def __init__(self, hazy_dir, clear_dir, patch_size):

        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.files = sorted(os.listdir(hazy_dir))
        self.patch_size = patch_size

    def set_patch(self, size):
        self.patch_size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        hazy = Image.open(os.path.join(self.hazy_dir, self.files[idx])).convert("RGB")
        clear = Image.open(os.path.join(self.clear_dir, self.files[idx])).convert("RGB")

        hazy = TF.to_tensor(hazy)
        clear = TF.to_tensor(clear)

        H, W = hazy.shape[1:]

        ps = self.patch_size

        r = random.randint(0, H-ps)
        c = random.randint(0, W-ps)

        hazy = hazy[:, r:r+ps, c:c+ps]
        clear = clear[:, r:r+ps, c:c+ps]

        if random.random() < 0.5:
            hazy = torch.flip(hazy, [2])
            clear = torch.flip(clear, [2])

        return hazy, clear