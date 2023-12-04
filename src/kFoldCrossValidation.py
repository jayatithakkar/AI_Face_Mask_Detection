import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
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

def train_validate_model(model, train_loader, val_loader, device, criterion, optimizer, num_epochs, patience, fold):
    best_val_loss = float('inf')
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            save_model(model, fold, 'best')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
    save_model(model, fold, 'final')

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

def save_model(model, fold, model_type):
    filename = f"model_fold_{fold}_{model_type}.pth"
    torch.save(model.state_dict(), os.path.join("/content/drive/MyDrive/kfold-models", filename))

def evaluate_fold_performance(model, val_loader, device):
    all_targets, all_predictions = [], []
    model.eval()
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(all_targets, all_predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(all_targets, all_predictions, average='micro')
    return accuracy, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro

# Prepare dataset and loaders
data_path = '/content/FINAL-DATASET-MERGED'
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.Grayscale(1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
full_dataset = datasets.ImageFolder(data_path, transform=transform)
train_indices, test_indices = train_test_split(np.arange(len(full_dataset)), test_size=0.1, random_state=42)
train_dataset = Subset(full_dataset, train_indices)
test_loader = DataLoader(Subset(full_dataset, test_indices), batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
num_epochs = 20
patience = 3
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

fold_performance = {"accuracy": [], "precision_macro": [], "recall_macro": [], "f1_macro": [], "precision_micro": [], "recall_micro": [], "f1_micro": []}

for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
    print(f'FOLD {fold}')
    train_subset = Subset(train_dataset, train_ids)
    val_subset = Subset(train_dataset, val_ids)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    model = CustomCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    train_validate_model(model, train_loader, val_loader, device, criterion, optimizer, num_epochs, patience, fold)

    metrics = evaluate_fold_performance(model, val_loader, device)
    for i, key in enumerate(fold_performance.keys()):
        fold_performance[key].append(metrics[i])

# Calculate and print average metrics
for metric, values in fold_performance.items():
    print(f'Average {metric}: {np.mean(values)}')

np.save('/content/drive/MyDrive/fold_performance.npy', fold_performance)
