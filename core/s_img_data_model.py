#!/usr/bin/env python3
"""
core/s_img_data_model.py

Trains two models:
  1. Random Forest on tabular fluorescence data
  2. CNN on synthetic fluorescence images

Optional: Can call with `emit_epoch` and `emit_batch` callbacks for GUI live plotting.
"""
import os
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

from core.plot_run_graph import plot_training_history_torch


# =====================================================
# Dataset Class
# =====================================================
class SporeDataset(Dataset):
    def __init__(self, folder, transform=None, img_size=128):
        self.transform = transform
        self.data, self.labels = [], []
        self.img_size = img_size

        for label, class_name in enumerate(["spores", "not_spores"]):
            class_path = os.path.join(folder, class_name)
            if not os.path.exists(class_path):
                print(f"⚠ Warning: missing folder {class_path}")
                continue
            for img_name in os.listdir(class_path):
                self.data.append(os.path.join(class_path, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.expand_dims(img, axis=0)  # (1, H, W)

        if self.transform:
            img = self.transform(torch.tensor(img, dtype=torch.float32))

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


# =====================================================
# CNN Model Definition
# =====================================================
class CNN(nn.Module):
    def __init__(self, img_size=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (img_size // 4) * (img_size // 4), 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


# =====================================================
# Training Pipeline
# =====================================================
def train_models(
    data_csv,
    image_dir,
    epochs,
    batch_size,
    save_model_dir,
    lr=3e-4,
    img_size=128,
    emit_epoch=None,  # optional callback: emit_epoch(epoch, loss, acc)
    emit_batch=None   # optional callback: emit_batch(global_percent)
):
    """
    Train Random Forest on tabular data and CNN on image dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    Path(save_model_dir).mkdir(parents=True, exist_ok=True)

    # --- Random Forest ---
    data = pd.read_csv(data_csv)
    X = data[['initial_amount', 'sample_concentration']].values
    y = data['spore_count'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    preds = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Random Forest RMSE: {rmse:.2f}")

    rf_model_path = os.path.join(save_model_dir, "rf_model.pkl")
    joblib.dump(rf_model, rf_model_path)
    print(f"Saved Random Forest model → {rf_model_path}")

    # --- CNN ---
    transform = transforms.Compose([transforms.Lambda(lambda x: x / 255.0)])
    dataset = SporeDataset(image_dir, transform=transform, img_size=img_size)
    if len(dataset) == 0:
        print("No images found. Skipping CNN training.")
        return {"rf_model": rf_model_path, "rf_rmse": rmse}

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    cnn = CNN(img_size=img_size).to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_losses, train_accs = [], []

    print(f"Training CNN for {epochs} epochs...")
    for epoch in range(epochs):
        cnn.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = cnn(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # --- Emit batch progress for GUI ---
            if emit_batch:
                global_percent = ((epoch + batch_idx / len(train_loader)) / epochs) * 100
                emit_batch(global_percent)

        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        train_losses.append(avg_loss)
        train_accs.append(acc)

        print(f"    Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")

        # --- Emit epoch for GUI ---
        if emit_epoch:
            emit_epoch(epoch + 1, avg_loss, acc)

    cnn_path = os.path.join(save_model_dir, "cnn_model.pth")
    torch.save(cnn.state_dict(), cnn_path)
    print(f"Saved CNN model @ {cnn_path}")

    # --- Save final training plot ---
    plot_training_history_torch(train_losses, train_accs, "spore_run_results.png")
    print("Training history saved @ spore_run_results.png")

    return {
        "rf_model": rf_model_path,
        "cnn_model": cnn_path,
        "rf_rmse": rmse,
        "epochs": epochs,
        "final_cnn_acc": train_accs[-1] if train_accs else None,
    }


# =====================================================
# CLI Entry Point
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Train CNN and Random Forest models on fluorescence data.")
    parser.add_argument("--data_csv", type=str, default="s_data.csv", help="Path to CSV data file")
    parser.add_argument("--image_dir", type=str, default="s_images", help="Path to image dataset folder")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs for CNN")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for CNN training")
    parser.add_argument("--save_model", type=str, default="models", help="Directory to save models")
    args = parser.parse_args()

    train_models(
        data_csv=args.data_csv,
        image_dir=args.image_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_model_dir=args.save_model,
    )


if __name__ == "__main__":
    main()
