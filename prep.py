import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_data_loader(root="Dataset", batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    dataset = ImageFolder(root=root, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader, dataset.classes

if __name__ == "__main__":
    loader, classes = get_data_loader()
    print("Dataset loaded with classes:", classes)
