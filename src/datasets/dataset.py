import os
import torch
import cv2
from torch.utils.data import Dataset

class MVTechDataset(Dataset):
    def __init__(self,root,split="train",img_size=224):
        self.labels=[]
        self.img_paths = []
        self.img_size = img_size
        base = os.path.join(root,split)

        for def_type in os.listdir(base):
            def_dir = os.path.join(base,def_type)
            for img in os.listdir(def_dir):
                self.img_paths.append(os.path.join(def_dir,img))
                self.labels.append(0 if def_type == "good" else 1)
                            
        return

    def __len__(self):return len(self.img_paths)

    def __getitem__(self,idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(self.img_size,self.img_size))
        img = img.astype("float32")/255.0
        img = torch.from_numpy(img).permute(2,0,1)
        return img, self.labels[idx]
    
