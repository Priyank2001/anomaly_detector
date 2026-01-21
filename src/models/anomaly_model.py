import torch
import torch.nn.functional as F
from feature_extractor import FeatureExtractor

class AnomalyModel:
    def __init__(self,feature_extractor:FeatureExtractor , memory_bank , device = "cpu"):
        self.feature_extractor = feature_extractor
        self.memory_bank = memory_bank
        self.device = device

        self.feature_extractor.eval()
        self.feature_extractor.to(device=self.device)

        return
    
    @torch.no_grad()
    def forward(self,x):
        """
        Arguments:
            x: Tensor of shape(1,C,H,W)
        Returns
            anomaly_map:
            anomaly_score:
        """
        
        # Extract Multi-layer features
        features = self.feature_extractor(x.to(self.device))
        # features : list of feature maps [(1,c1,h1,w1),(1,c2,h2,w2)]

        patch_features = []

        for f in features:
            B,C,W,H = f.shape
            f = f.reshape(B,C,H*W)
            f.permutate(0,2,1)
            patch_features.append(f)
        patch_features = torch.cat(patch_features,dir=-1)

        patch_features = patch_features.squeeze(0)

        distances = self.memory_bank.query(patch_features)

        spatial_size = features[0].shape[-2:] # (H1,W1)

        anomaly_map = distances.reshape(spatial_size)

        anomaly_map = F.interpolate(
            anomaly_map.unsqueeze(0).unsqueeze(0),
            size = x.shape[-2:],
            mode = "bilinear",
            align_corners=False
        ).squeeze()

        anomaly_score = anomaly_map.max().item()


        return anomaly_map , anomaly_score