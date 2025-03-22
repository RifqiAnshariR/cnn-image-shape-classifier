import torch
from train import ShapeClassifier
from prep import get_data_loader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

def load_model(model_path="./model/shape_classifier.pth"):
    train_loader, classes = get_data_loader()
    model = ShapeClassifier(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, classes

def predict(image_path):
    model, classes = load_model()
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return classes[predicted.item()], confidence.item()

if __name__ == "__main__":
    img_path = "./Test_Data/circle_2.png"
    label, conf = predict(img_path)
    print(f"Predicted shape: {label} (Confidence: {conf:.2%})")
