import torch
import numpy as np
from torch.utils.data import DataLoader
from src.datasets.dataset import MVTechDataset
from feature_extractor import FeatureExtractor

import torch.nn.functional as F

device = "cpu"

dataset = MVTechDataset(root="../../data/bottle",split="train")

loader = DataLoader(dataset=dataset,batch_size=8,shuffle=False)

model = FeatureExtractor().to(device)
model.eval()

memoryBank = []

with torch.no_grad():
    for imgs,_ in loader:
        imgs = imgs.to(device)
        f2,f3 = model(imgs)

        f2 = F.interpolate(
            f2,
            size=f3.shape[-2:],   # (14, 14)
            mode="bilinear",
            align_corners=False
        )

        f2 = f2.permute(0,2,3,1).contiguous()
        f2 =f2.view(-1,f2.shape[-1])
        f3 = f3.permute(0,2,3,1).contiguous()
        f3 = f3.view(-1,f3.shape[-1])

        features = torch.cat([f2,f3],dim = 1)

        memoryBank.append(features.cpu().numpy())

memoryBank = np.concatenate(memoryBank,axis=0)

np.save("memory_bank.npy",memoryBank)
print("Memory Bank saved:",memoryBank.shape)
