import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class CustomCNN(nn.Module):
    """
    A custom CNN model for image classification.

    The network consists of five convolutional layers with batch normalization and dropout,
    followed by three fully connected layers. The model uses ReLU activation and max pooling.

    Args:
        num_classes (int): Number of classes for the output layer.

    Returns:
        A CustomCNN object.
    """

    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout2 = nn.Dropout(0.25)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.25)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256*8*8, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn7 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.dropout1(self.bn2(self.conv2(x)))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.dropout2(self.bn4(self.conv4(x)))))
        x = self.pool(F.relu(self.dropout3(self.bn5(self.conv5(x)))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.bn6(self.fc1(x)))
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def load_model(model_path, device, num_classes=4):
    """
    Load the trained model from the specified path.

    Args:
        model_path (str): Path to the saved model.
        device (torch.device): Device to load the model onto.
        num_classes (int, optional): Number of classes for the output layer.

    Returns:
        nn.Module: Loaded model.
    """
    model = CustomCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(image_path, model, device, class_names):
    """
    Predict the class of an input image using the provided model.

    Args:
        image_path (str): Path to the input image.
        model (nn.Module): Trained model for prediction.
        device (torch.device): Device to perform prediction on.
        class_names (list): List of class names.

    Returns:
        torch.Tensor: Processed image tensor.
        str: Predicted class label.
    """
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load and transform the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    return image.cpu().squeeze(0), predicted_class

def show_image_with_prediction(image, predicted_class):
    """
    Display the image along with the predicted class label.

    Args:
        image (torch.Tensor): Image tensor to display.
        predicted_class (str): Predicted class label.
    """
    plt.imshow(image.permute(1, 2, 0), cmap='gray')
    plt.title(f'Predicted Class: {predicted_class}')
    plt.axis('off')
    plt.show()

def main():
    """
    Main function to demonstrate image prediction using the trained model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '../model/variant2-latest.pth'
    image_path = 'image.jpg'
    class_names = ['angry', 'bored', 'focused', 'neutral']

    model = load_model(model_path, device)
    image, predicted_class = predict_image(image_path, model, device, class_names)
    show_image_with_prediction(image, predicted_class)

if __name__ == '__main__':
    main()
