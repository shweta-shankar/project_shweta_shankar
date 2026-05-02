import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from config import resize_x, resize_y, batch_size, val_split, num_workers, mean, std

def get_dataloaders(data_dir, seed=42):
    train_transforms = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    class_names = full_dataset.classes

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(seed))

    val_dataset = datasets.ImageFolder(root=data_dir, transform=val_test_transforms)
    val_ds = Subset(val_dataset, val_ds.indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
                            
    return train_loader, val_loader, class_names

# Expose a custom Dataset class for interface.py
class ASLDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
