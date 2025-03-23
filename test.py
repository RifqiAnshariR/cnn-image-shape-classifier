import torch
import torch.nn.functional as F
from train import ShapeClassifier
from prep import get_data_loaders
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report

def load_model(model_path="./Model/shape_classifier.pth"):
    _, _, classes = get_data_loaders()
    model = ShapeClassifier(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, classes

def predict(model, classes, image_path):
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

    return classes[predicted.item()], confidence.item() * 100

def evaluate_model(model, test_loader, classes):
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=classes))

if __name__ == "__main__":
    model, classes = load_model()
    _, test_loader, _ = get_data_loaders()

    # Evaluasi model
    evaluate_model(model, test_loader, classes)

    # Prediksi contoh gambar
    img_path = "./Test_Data/triangle_1.png"
    shape, confidence = predict(model, classes, img_path)
    print(f"Predicted shape: {shape} with confidence: {confidence:.2f}%")
