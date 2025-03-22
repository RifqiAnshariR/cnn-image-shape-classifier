import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from prep import get_data_loader

class ShapeClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 32 * 32, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_model(epochs=50, lr=0.001):
    train_loader, classes = get_data_loader()
    model = ShapeClassifier(num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} selesai")

    torch.save(model.state_dict(), "./model/shape_classifier.pth")
    print("Model saved!")

if __name__ == "__main__":
    train_model()
