import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class CustomCNN(nn.Module):
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
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 128)
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
    model = CustomCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def create_dataloader(data_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = datasets.ImageFolder(data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size)

def evaluate_model(model, loader, device):
    all_targets, all_predictions = [], []
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    model.eval()
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')
    metrics["accuracy"].append(accuracy)
    metrics["precision"].append(precision)
    metrics["recall"].append(recall)
    metrics["f1"].append(f1)

    return metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './experiment1_fold_1_best.pth'
model = load_model(model_path, device)

# Create data loaders
data_paths = ['/content/BIAS-AGE-DATASET-NEW/young', '/content/BIAS-AGE-DATASET-NEW/middle-aged', '/content/BIAS-AGE-DATASET-NEW/senior']
loaders = [create_dataloader(path) for path in data_paths]

# Evaluate model and print metrics
for loader in loaders:
    metrics = evaluate_model(model, loader, device)
    print(metrics)

# Store the results
results = [evaluate_model(model, loader, device) for loader in loaders]
np.save('/content/drive/MyDrive/age_based_performance_v1.npy', results)
