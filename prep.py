import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

def get_data_loaders(root, batch_size, split_ratio):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    dataset = ImageFolder(root=root, transform=transform)

    # Split dataset train (80%) dan test (20%)
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset.classes

if __name__ == "__main__":
    train_loader, test_loader, classes = get_data_loaders("Dataset", 32, 0.8)
    print("Dataset loaded with classes:", classes)
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
