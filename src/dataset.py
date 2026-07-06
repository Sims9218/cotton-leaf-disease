import os
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class RobustLeafDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        if not os.path.isdir(root_dir):
            raise ValueError(f"Directory {root_dir} does not exist")

        self.classes = sorted([d for d in os.listdir(root_dir)
                             if os.path.isdir(os.path.join(root_dir, d))])
        if not self.classes:
            raise ValueError(f"No valid class directories found in {root_dir}")

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._build_samples()

        if not self.samples:
            raise ValueError(f"No valid images found in {root_dir}")

    def _build_samples(self):
        samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_dir, img_name)
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        samples.append((img_path, self.class_to_idx[cls]))
                    except Exception as e:
                        print(f"Skipping corrupt image {img_path}: {str(e)}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            img_path, label = self.samples[idx]
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
                    if img is None:
                        raise ValueError("Transform returned None")
                return img, label
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return torch.zeros(3, 224, 224), 0
