"""
Load and prepare Fashion-MNIST dataset with augmentation
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CLASSES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def get_loaders(batch_size=64):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_data = datasets.FashionMNIST(
        root='data', train=True, download=True, transform=train_transform
    )
    test_data = datasets.FashionMNIST(
        root='data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_data)} images | Test: {len(test_data)} images")
    return train_loader, test_loader
