import torch
from datasets.dataset import MVTecBottleDataset
from feature_extractor import FeatureExtractor
from memory_bank import MemoryBank
from anomaly_model import AnomalyModel

# ------------------
# CONFIG
# ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "data/mvtec/bottle"
IMAGE_SIZE = 224

# ------------------
# 1. Load ONE training image (normal)
# ------------------
train_dataset = MVTecBottleDataset(
    root=DATA_ROOT,
    split="train",
)

img, _, _ = train_dataset[0]
img = img.unsqueeze(0).to(DEVICE)  # (1, C, H, W)

print("Input image shape:", img.shape)

# ------------------
# 2. Initialize feature extractor
# ------------------
feature_extractor = FeatureExtractor()
feature_extractor.to(DEVICE)

# ------------------
# 3. Build memory bank from ONE image (sanity only)
# ------------------
memory_bank = MemoryBank()

with torch.no_grad():
    feats = feature_extractor(img)
    patch_features = []

    for f in feats:
        B, C, H, W = f.shape
        f = f.reshape(B, C, H * W).permute(0, 2, 1)
        patch_features.append(f)

    patch_features = torch.cat(patch_features, dim=-1)
    patch_features = patch_features.squeeze(0)

memory_bank.add(patch_features)

print("Memory bank size:", memory_bank.features.shape)

# ------------------
# 4. Initialize anomaly model
# ------------------
model = AnomalyModel(feature_extractor, memory_bank, device=DEVICE)

# ------------------
# 5. Forward pass
# ------------------
anomaly_map, anomaly_score = model.forward(img)

print("Anomaly map shape:", anomaly_map.shape)
print("Anomaly score:", anomaly_score)
