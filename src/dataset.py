# dataset.py

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_datasets(
    data_dir: str, 
    image_size: int = 224, 
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42
) -> Tuple[Subset, Subset, Subset, List[str]]:
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    train_ratio, val_ratio, test_ratio = split_ratios

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    full_dataset_train = datasets.ImageFolder(data_dir, transform=train_transform)
    full_dataset_eval = datasets.ImageFolder(data_dir, transform=eval_transform)

    class_names = full_dataset_train.classes
    targets = full_dataset_train.targets
    indices = list(range(len(targets)))

    train_indices, temp_indices, _, temp_targets = train_test_split(
        indices,
        targets,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=targets
    )

    relative_test_size = test_ratio / (val_ratio + test_ratio)
    
    val_indices, test_indices, _, _ = train_test_split(
        temp_indices,
        temp_targets,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_targets
    )


    train_dataset = Subset(full_dataset_train, train_indices)
    val_dataset = Subset(full_dataset_eval, val_indices)
    test_dataset = Subset(full_dataset_eval, test_indices)

    return train_dataset, val_dataset, test_dataset, class_names


def get_dataloaders(
    data_dir: str, 
    batch_size: int = 32, 
    image_size: int = 224, 
    num_workers: Optional[int] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    
    train_dataset, val_dataset, test_dataset, class_names = get_datasets(
        data_dir, image_size, seed=seed
    )

    if num_workers is None:
        num_workers = min(4, os.cpu_count() if os.cpu_count() else 1)
       
    g = torch.Generator()
    g.manual_seed(seed)

    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0), 
    )

    train_loader = DataLoader(train_dataset, shuffle=True, generator=g, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader, class_names