import torch

from src.datasets.dataset import MVTechDataset
from src.models.feature_extractor import FeatureExtractor
from src.models.memory_bank import MemoryBank
from src.models.anomaly_model import AnomalyModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "./data/bottle"

# ------------------
# Load ONE image
# ------------------
dataset = MVTechDataset(root=DATA_ROOT, split="train")
img , _ = dataset[0]
img = img.unsqueeze(0).to(DEVICE)


# ------------------
# Feature extractor
# ------------------
feature_extractor = FeatureExtractor().to(DEVICE)
feature_extractor.eval()


# ------------------
# Build memory bank (sanity test)
# ------------------
memory_bank = MemoryBank()


with torch.no_grad():
    f2, f3 = feature_extractor(img)
    B, C2, H2, W2 = f2.shape
    B, C3, H3, W3 = f3.shape

    # Upsample f3 to match spatial dimensions of f2 before concatenation
    if (H2, W2) != (H3, W3):
        import torch.nn.functional as F
        f3 = F.interpolate(f3, size=(H2, W2), mode="bilinear", align_corners=False)
    f2 = f2.reshape(B, C2, -1).permute(0, 2, 1)
    f3 = f3.reshape(B, C3, -1).permute(0, 2, 1)
    patch_features = torch.cat([f2, f3], dim=-1).squeeze(0)
memory_bank.add(patch_features)
print("Memory bank size:", memory_bank.features.shape)

# ------------------
# Anomaly model
# ------------------
model = AnomalyModel(feature_extractor, memory_bank, device=DEVICE)
anomaly_map, anomaly_score = model(img)

print("Anomaly map shape:", anomaly_map.shape)
print("Anomaly score:", anomaly_score)
