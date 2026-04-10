import numpy as np
import torch
from torch.utils.data import Dataset


class NDataSet(Dataset):
    def __init__(self, mris, labels, pets, stas=None, aug=None, norm=False):
        super().__init__()
        self.mris = mris
        self.labels = labels
        self.pets = pets
        self.stas = stas
        self.aug = aug
        self.norm = norm

    def __getitem__(self, index):
        img = self.mris[index]
        img2 = self.pets[index]
        lbl = self.labels[index]
        mmse = float(self.stas[index]) / 30.0
        if self.norm:
            img = img / np.max(img)
            img2 = img2 / np.max(img2)
        if self.aug:
            for aug in self.aug:
                img = aug(img)
                img2 = aug(img2) if img2 is not None else None
        img = torch.as_tensor(img, dtype=torch.float32).unsqueeze(dim=0)  # dhw->cdhw
        img2 = torch.as_tensor(img2, dtype=torch.float32).unsqueeze(dim=0) if img2 is not None else None  # dhw->cdhw
        lbl = torch.as_tensor(lbl, dtype=torch.long)
        mmse = torch.tensor(mmse, dtype=torch.float32)
        return img, lbl, img2, mmse

    def __len__(self):
        return len(self.labels)
