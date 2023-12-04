import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class CustomCNN(nn.Module):
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
    model = CustomCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

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

def create_dataloader(path, transform, batch_size=32):
    dataset = datasets.ImageFolder(path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size)

# Define transformations and create data loaders
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

male_loader = create_dataloader('/content/BIAS-GENDER-DATASET-NEW/Male', transform)
female_loader = create_dataloader('/content/BIAS-GENDER-DATASET-NEW/Female', transform)

# Load model and evaluate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('./experiment1_fold_1_best.pth', device)
male_metrics = evaluate_model(model, male_loader, device)
female_metrics = evaluate_model(model, female_loader, device)

# Print and save the results
print(male_metrics)
print(female_metrics)
results = [male_metrics, female_metrics]
np.save('/content/drive/MyDrive/gender_based_performance_v1.npy', results)
