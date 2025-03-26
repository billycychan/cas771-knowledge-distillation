import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os

class CAS771Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            else:
                img = transforms.ToPILImage()(img)
            img = self.transform(img)
        return img, label

def load_data(data_path):
    raw_data = torch.load(data_path)
    data = raw_data['data'].numpy()
    labels = raw_data['labels'].numpy()
    return data, labels

def remap_labels(labels, class_mapping):
    return [class_mapping[label] for label in labels]

def calculate_normalization_stats(dataloader):
    """Calculate channel-wise mean and std for a dataset"""
    # Accumulate values
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    num_pixels = 0

    # Process all images
    for images, _ in dataloader:
        channel_sum += torch.mean(images, dim=[0,2,3]) * images.size(0)
        channel_sum_sq += torch.mean(images ** 2, dim=[0,2,3]) * images.size(0)
        num_pixels += images.size(0)

    # Calculate mean and std
    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sum_sq / num_pixels - mean ** 2)

    return mean, std

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        layers = [ConvBlock(in_channels, out_channels)]
        for _ in range(num_convs - 1):
            layers.append(ConvBlock(out_channels, out_channels))
        self.main_path = nn.Sequential(*layers)
        # self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return F.relu(self.main_path(x) + self.shortcut(x))
    
class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = ConvBlock(3, 64)

        self.residual1 = ResidualBlock(64, 128)
        self.residual2 = ResidualBlock(128, 256, num_convs=2)

        self.conv2 = ConvBlock(256, 512, kernel_size=1, padding=0)

        # 使用 Depthwise Separable Convolution 代替普通 3x3 卷积
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(512, 512),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual1(x)
        x = F.max_pool2d(x, 2)
        x = self.residual2(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(self.conv3(x))

        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def load_and_prepare_data(model_used, batch_size=256):
    data_paths = {
        1: ('./data/TaskB/train_dataB_model_1.pth',
            './data/TaskB/val_dataB_model_1.pth'),
        2: ('./data/TaskB/train_dataB_model_2.pth',
            './data/TaskB/val_dataB_model_2.pth'),
        3: ('./data/TaskB/train_dataB_model_3.pth',
            './data/TaskB/val_dataB_model_3.pth')
    }

    train_data_path, test_data_path = data_paths[model_used]

    # train_data, train_labels, _ = load_data(train_data_path)
    train_data, train_labels = load_data(train_data_path)
    # test_data, test_labels, _ = load_data(test_data_path)
    test_data, test_labels = load_data(test_data_path)

    unique_labels = sorted(set(train_labels))
    class_mapping = {label: i for i, label in enumerate(unique_labels)}

    mapped_train_labels = remap_labels(train_labels, class_mapping)
    mapped_test_labels = remap_labels(test_labels, class_mapping)

    initial_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = CAS771Dataset(train_data, mapped_train_labels, transform=initial_transform)
    test_dataset = CAS771Dataset(test_data, mapped_test_labels, transform=initial_transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    cal_norm = True
    if cal_norm:
      mean, std = calculate_normalization_stats(train_loader)
      norm_transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean=mean.tolist(), std=std.tolist())
      ])

      train_dataset = CAS771Dataset(train_data, mapped_train_labels, transform=norm_transform)
      test_dataset = CAS771Dataset(test_data, mapped_test_labels, transform=norm_transform)

      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
      test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, len(unique_labels)

def train_and_evaluate(model, train_loader, test_loader, num_epochs=100, lr=0.0005, weight_decay=5e-4, step_size=1, gamma=0.8, model_used=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_losses, train_accuracies, test_accuracies = [], [], []
    best_test_acc = 0

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = 100.0 * correct_test / total_test
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_test_acc:
            
            if test_acc > 80 and test_acc < 83:
                scheduler.step()
                print(f"Learning rate after step: {optimizer.param_groups[0]['lr']:.10f}")
                
            elif test_acc > 83 and test_acc < 84:
                scheduler.step(10)
                print(f"Learning rate after step: {optimizer.param_groups[0]['lr']:.10f}")
                
            elif test_acc > 84:
                scheduler.step(30)
                print(f"Learning rate after step: {optimizer.param_groups[0]['lr']:.10f}")
                
                
            for file in os.listdir("./models/B"):
                if file.startswith(f"data{model_used}_"):
                    os.remove(os.path.join("./models/B",file))

            torch.save(model.state_dict(), f'./models/B/data{model_used}_{test_acc:.2f}_epoch{epoch}.pth')
            print("Checkpoint saved to"+ f'./models/B/data{model_used}_{test_acc:.2f}_epoch{epoch}.pth.')
            
            best_test_acc = test_acc
            
        test_accuracies.append(test_acc)

    plot_learning_curve(num_epochs, train_losses, train_accuracies, test_accuracies)
    
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def plot_learning_curve(num_epochs, train_losses, train_accuracies, test_accuracies):
    def smooth_curve(values, window=5, poly=2):
        return savgol_filter(values, window, poly)

    epochs = range(1, num_epochs + 1)
    smoothed_train_losses = smooth_curve(train_losses)
    smoothed_train_accuracies = smooth_curve(train_accuracies)
    smoothed_test_accuracies = smooth_curve(test_accuracies)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, smoothed_train_losses, label='Training Loss', linestyle='-', linewidth=2, color='tab:red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, smoothed_train_accuracies, label='Training Accuracy', linestyle='-', linewidth=2, color='tab:blue')
    plt.plot(epochs, smoothed_test_accuracies, label='Test Accuracy', linestyle='-', linewidth=2, color='tab:green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to: {seed}")


if __name__ == "__main__":
    set_seed(100)

    train_loader, test_loader, num_classes = load_and_prepare_data(model_used=1, batch_size=32)
    model = CNN(num_classes)
    train_and_evaluate(model, train_loader, test_loader, num_epochs=200, model_used=1)

