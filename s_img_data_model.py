import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from torchvision import models
from plot_run_graph import plot_training_history_torch

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Experimental Data (Simulated Dataset Example)
data = pd.read_csv('s_data.csv')

# Extract Features & Labels for Random Forest
X = data[['initial_amount', 'sample_concentration']].values
y = data['spore_count'].values  # Target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Model for Predicting spore count
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict & Evaluate
predictions = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'Random Forest RMSE: {rmse:.2f}')


# PyTorch Dataset & DataLoader
# ==============================
IMG_SIZE = 128

# Define Custom Dataset
class SporeDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []

        for label, class_name in enumerate(["spores", "not_spores"]):
            class_path = os.path.join(folder, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.data.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img, axis=0)  # Add channel dimension

        if self.transform:
            img = self.transform(torch.tensor(img, dtype=torch.float32))

        return img, torch.tensor(label, dtype=torch.float32)

# Define Transformations
transform = transforms.Compose([
    transforms.Lambda(lambda x: x / 255.0),  # Normalize images
])

# Load datasets
dataset = SporeDataset("s_images", transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define CNN Model in PyTorch
# ==============================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid for binary classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# Instantiate the model
cnn_model = CNN().to(device)
optimizer = optim.Adam(cnn_model.parameters(), lr=0.0003)
criterion = nn.BCELoss()


# Train CNN Model
# ==============================

# Initialize lists to store loss and accuracy
train_losses = []
train_accuracies = []

EPOCHS = 10

for epoch in range(EPOCHS):
    cnn_model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = cnn_model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Store loss and accuracy
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")


# Evaluate CNN Performance
# ==============================
cnn_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = cnn_model(images).squeeze()
        predicted = (outputs > 0.5).float()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to NumPy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Count correctly classified images for each category
correct_spores = np.sum((all_preds == 0) & (all_labels == 0))  # True Positives (spores classified as spores)
correct_not_spores = np.sum((all_preds == 1) & (all_labels == 1))  # True Negatives (not_spores classified as not_spores)

# Count misclassified images
wrong_spores = np.sum((all_preds == 1) & (all_labels == 0))  # False Negatives (spores misclassified as not_spores)
wrong_not_spores = np.sum((all_preds == 0) & (all_labels == 1))  # False Positives (not_spores misclassified as spores)

# Create summary DataFrame
summary_table = pd.DataFrame({
    "Category": ["Spores", "Not Spores"],
    "Correctly Classified": [correct_spores, correct_not_spores],
    "Misclassified": [wrong_spores, wrong_not_spores],
    "Total": [correct_spores + wrong_spores, correct_not_spores + wrong_not_spores],
    "Accuracy (%)": [
        100 * correct_spores / (correct_spores + wrong_spores) if (correct_spores + wrong_spores) > 0 else 0,
        100 * correct_not_spores / (correct_not_spores + wrong_not_spores) if (correct_not_spores + wrong_not_spores) > 0 else 0
    ]
})

# Print summary table
print("\nClassification Summary:")
print(summary_table)

plot_training_history_torch(train_losses, train_accuracies, "spore_run_results.png")

# # Compute accuracy
# accuracy = accuracy_score(all_labels, all_preds)
# cm = confusion_matrix(all_labels, all_preds)

# print(f'CNN Accuracy: {accuracy * 100:.2f}%')
# sns.heatmap(cm, annot=True, fmt='d')
# plt.show()
