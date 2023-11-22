import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class CustomCNN(nn.Module):
    """
    A custom CNN model for image classification.

    The network consists of four convolutional layers with batch normalization and dropout,
    followed by three fully connected layers. The model uses ReLU activation and max pooling.

    Args:
        num_classes (int): Number of classes for the output layer.

    Returns:
        A CustomCNN object.
    """

    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout2 = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 16 * 16, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.dropout1(self.bn2(self.conv2(x)))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.dropout2(self.bn4(self.conv4(x)))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn5(self.fc1(x)))
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def load_model(model_path, device, num_classes=4):
    """
    Load a pre-trained model from a given path and return it.

    Parameters:
        model_path (str): The path to the saved model file.
        device (torch.device): The device to load the model on.
        num_classes (int, optional): The number of classes in the model's output. Defaults to 4.

    Returns:
        CustomCNN: The loaded model.
    """
    model = CustomCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    return model

def load_data(data_path, batch_size=64):
    """
    Load data from the given data_path and create a DataLoader object.

    Args:
        data_path (str): The path to the directory containing the data.
        batch_size (int, optional): The number of samples per batch. Defaults to 64.

    Returns:
        DataLoader: The DataLoader object containing the loaded data.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = ImageFolder(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

def evaluate(model, test_loader, device):
    """
    Evaluate the model on the test set.

    Args:
        model (CustomCNN): The model to evaluate.
        test_loader (DataLoader): The DataLoader containing the test set.
        device (torch.device): The device to use for evaluation.

    Returns:
        tuple: A tuple containing the test accuracy, confusion matrix, and classification report.
    """
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    predictions, true_labels = [], []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, targets).item()
            preds = outputs.argmax(1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    test_accuracy = 100 * correct / total
    confusion = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions, output_dict=True)

    return test_accuracy, confusion, report

def plot_confusion_matrix(confusion, class_names):
    """
    Plots a confusion matrix using Seaborn.

    Args:
        confusion (numpy.ndarray): The confusion matrix to plot.
        class_names (list): The names of the classes.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_classification_report(report):
    """
    Plots a classification report using Matplotlib.

    Args:
        report (dict): The classification report to plot.

    Returns:
        None
    """
    report = report['macro avg']
    report_df = pd.DataFrame(report).transpose()
    report_df.drop('support', axis=1, inplace=True)
    report_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Classification Report')
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.xticks(rotation=45)
    plt.show()

def main():
    """
    Main function to evaluate the model on the test set.

    Returns:
        None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '../model/mainmodel-latest.pth'
    test_data_path = '/content/DATASET/test'
    test_labels_path = '/content/DATASET/test/test_labels.csv'

    model = load_model(model_path, device)
    test_loader = load_data(test_data_path)
    test_accuracy, confusion, report = evaluate(model, test_loader, device)

    print(f'Test Accuracy: {test_accuracy:.2f}%')

    test_labels = pd.read_csv(test_labels_path)
    label_encoder = LabelEncoder().fit(test_labels['Label'])
    plot_confusion_matrix(confusion, label_encoder.classes_)
    plot_classification_report(report)

if __name__ == '__main__':
    main()
