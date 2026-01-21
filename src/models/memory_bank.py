import torch

class MemoryBank:
    def __init__(self):
        self.features = None

    def add(self, features: torch.Tensor):
        """
        features: (N, D)
        """
        if self.features is None:
            self.features = features
        else:
            self.features = torch.cat([self.features, features], dim=0)

    def size(self):
        return 0 if self.features is None else self.features.shape[0]

    def query(self, features: torch.Tensor):
        """
        Query the memory bank with `features`.

        features: Tensor of shape (N, D)
        Returns: distances Tensor of shape (N,) with the minimal euclidean distance
        from each query feature to the memory bank features.
        """
        if self.features is None:
            raise RuntimeError("Memory bank is empty. Add features before querying.")

        # Ensure tensors are on the same device
        bank = self.features.to(features.device)

        # Compute pairwise distances and take min across bank entries
        # cdist returns shape (N, M)
        dists = torch.cdist(features, bank)
        mins, _ = dists.min(dim=1)
        return mins
