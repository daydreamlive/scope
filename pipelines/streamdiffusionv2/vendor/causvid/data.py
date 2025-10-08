from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb

import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange
import os


class TextDataset(Dataset):
    def __init__(self, data_path):
        self.texts = []
        with open(data_path, "r") as f:
            for line in f:
                self.texts.append(line.strip())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
