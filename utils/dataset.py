import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SyntheticDetectionDataset(Dataset):
    """
    High-performance synthetic dataset generator for pipeline verification
    and benchmarking without needing a 20GB COCO download immediately.
    """
    def __init__(self, num_samples=100, img_size=640, num_classes=80):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic image tensor [3, H, W] normalized 0-1
        img = torch.rand(3, self.img_size, self.img_size, dtype=torch.float32)

        # Generate synthetic targets: [class_id, x_center, y_center, width, height]
        num_boxes = np.random.randint(1, 5)
        boxes = []
        for _ in range(num_boxes):
            cls = np.random.randint(0, self.num_classes)
            cx, cy = np.random.uniform(0.2, 0.8, 2)
            w, h = np.random.uniform(0.05, 0.3, 2)
            boxes.append([cls, cx, cy, w, h])

        targets = torch.tensor(boxes, dtype=torch.float32)
        return img, targets

def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    # Add batch index to each target: [batch_idx, cls, cx, cy, w, h]
    new_targets = []
    for i, t in enumerate(targets):
        b_idx = torch.full((t.shape[0], 1), i, dtype=torch.float32)
        new_targets.append(torch.cat([b_idx, t], dim=1))
    new_targets = torch.cat(new_targets, 0)
    return imgs, new_targets

def build_dataloader(batch_size=4, num_workers=0):
    dataset = SyntheticDetectionDataset(num_samples=64, img_size=640)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    return loader