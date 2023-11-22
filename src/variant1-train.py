"""
This script defines and utilizes a convolutional neural network (CNN) for image classification.
It includes the CustomCNN class for the CNN model, functions for loading datasets, training and validating the model,
and evaluating its performance on a test dataset. The script is designed to work with a specific directory structure
for training, validation, and testing datasets.

Author: Dhruv Nareshkumar Panchal
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


class CustomCNN(nn.Module):
    """
    A custom CNN model for image classification.

    The network consists of three convolutional layers with batch normalization and dropout,
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
        self.dropout1 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256 * 32 * 32, 128)

        self.bn5 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.dropout1(self.conv2(x)))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.bn5(self.fc1(x)))
        x = F.relu(self.bn6(self.fc2(x)))

        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def load_dataset(data_path, transform):
    """
    Loads an image dataset from a specified path.

    Args:
        data_path (str): Path to the dataset directory.
        transform (transforms.Compose): Transformations to be applied to the dataset images.

    Returns:
        An ImageFolder dataset containing the loaded images and labels.
    """

    return ImageFolder(data_path, transform=transform)


def create_data_loaders():
    """
    Create data loaders for training, validation, and testing.

    Returns:
        train_loader (DataLoader): A data loader for the training dataset.
        val_loader (DataLoader): A data loader for the validation dataset.
        test_loader (DataLoader): A data loader for the testing dataset.
    """

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_dataset = load_dataset("/content/DATASET/train", transform)
    val_dataset = load_dataset("/content/DATASET/val", transform)
    test_dataset = load_dataset("/content/DATASET/test", transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    return train_loader, val_loader, test_loader


def train_and_validate(model, train_loader, val_loader, device, epochs=20, patience=5):
    """
    Trains and validates the model.

    The function runs training for a specified number of epochs, performing validation after each epoch.
    It implements early stopping based on the validation loss.

    Args:
        model (nn.Module): The CNN model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): The device to run the model on (CPU or GPU).
        epochs (int): Number of training epochs.
        patience (int): Patience for early stopping based on validation loss.

    Returns:
        None
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float("inf")
    counter = 0

    for epoch in range(epochs):
        model.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, targets).item()
                correct += (outputs.argmax(1) == targets).sum().item()
                total += targets.size(0)

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(
            f"Epoch {epoch+1}: Val Loss {val_loss:.4f}, Val Accuracy {val_accuracy:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    torch.save(model.state_dict(), "variant1-latest.pth")


def evaluate_model(model, test_loader, device):
    """
    Evaluates the model's performance on the test dataset.

    Calculates and prints the test accuracy, confusion matrix, and classification report.

    Args:
        model (nn.Module): The trained CNN model to evaluate.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
        None
    """

    criterion = nn.CrossEntropyLoss()
    test_loss, correct, total = 0.0, 0, 0
    predictions = []

    model.eval()
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            preds = outputs.argmax(1)
            predictions.extend(preds.cpu().numpy())
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    test_labels = pd.read_csv("/content/DATASET/test/test_labels.csv")
    label_encoder = LabelEncoder().fit(test_labels["Label"])
    true_labels = label_encoder.transform(test_labels["Label"])

    confusion = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(confusion)

    classification_rep = classification_report(
        true_labels, predictions, target_names=label_encoder.classes_
    )
    print("Classification Report:")
    print(classification_rep)
    model.train()


def main():
    """
    Main function to execute the model training and evaluation.

    Initializes the model, data loaders, and carries out the training, validation, and evaluation processes.

    Returns:
        None
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomCNN().to(device)

    train_loader, val_loader, test_loader = create_data_loaders()
    train_and_validate(model, train_loader, val_loader, device)
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
