import torch
import torch.nn.functional as F
from src.models.feature_extractor import FeatureExtractor

class AnomalyModel:
    def __init__(self,feature_extractor:FeatureExtractor , memory_bank , device = "cpu"):
        self.feature_extractor = feature_extractor
        self.memory_bank = memory_bank
        self.device = device

        self.feature_extractor.eval()
        self.feature_extractor.to(device=self.device)

        return
    def __call__(self, x):
        return self.forward(x)

    @torch.no_grad()
    def forward(self,x):
        """
        Arguments:
            x: Tensor of shape(1,C,H,W)
        Returns
            anomaly_map:
            anomaly_score:
        """
        
        # Extract Multi-layer features (tuple/list of feature maps)
        features = self.feature_extractor(x.to(self.device))
        if not isinstance(features, (list, tuple)):
            features = [features]

        # Determine target spatial size (use the largest H,W among features)
        spatial_sizes = [ (f.shape[-2], f.shape[-1]) for f in features ]
        target_H = max(s[0] for s in spatial_sizes)
        target_W = max(s[1] for s in spatial_sizes)

        patch_features_per_layer = []
        for f in features:
            B, C, H, W = f.shape
            if (H, W) != (target_H, target_W):
                import torch.nn.functional as F
                f = F.interpolate(f, size=(target_H, target_W), mode="bilinear", align_corners=False)

            # reshape to (B, H*W, C)
            f = f.reshape(B, C, -1).permute(0, 2, 1)
            patch_features_per_layer.append(f)

        # Concatenate channel dims for each spatial patch -> (B, num_patches, D)
        patch_features = torch.cat(patch_features_per_layer, dim=2)

        # remove batch dim (expects B==1)
        patch_features = patch_features.squeeze(0)

        # Query memory bank: returns distances for each patch (num_patches,)
        distances = self.memory_bank.query(patch_features)

        spatial_size = (target_H, target_W)

        anomaly_map = distances.reshape(spatial_size)

        anomaly_map = F.interpolate(
            anomaly_map.unsqueeze(0).unsqueeze(0),
            size = x.shape[-2:],
            mode = "bilinear",
            align_corners=False
        ).squeeze()

        anomaly_score = anomaly_map.max().item()


        return anomaly_map , anomaly_score
    
    def extract_patches():      
        
        return