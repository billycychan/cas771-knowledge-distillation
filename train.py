import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import random
import sys

# Import custom modules
from data_loader_kd import load_all_datasets, custom_collate_fn
from teacher_model_B import CNN, set_seed

class StudentModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Student model architecture - similar to teacher but smaller
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
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


def load_teacher_models(device, task='B'):
    """
    Load teacher models for each of the three datasets
    """
    teacher_models = {}
    
    # Define paths to teacher model weights based on task
    if task == 'A':
        model_paths = {
            1: './weights/A/data1_73.60_epoch151.pth',
            2: './weights/A/data2_80.60_epoch191.pth',
            3: './weights/A/data3_87.80_epoch184.pth'
        }
        # Original number of classes for each teacher model in Task A
        num_classes = {1: 5, 2: 5, 3: 5}
    else:  # task B
        model_paths = {
            1: './weights/B/data1_86.40_epoch182.pth',
            2: './weights/B/data2_92.80_epoch93.pth',
            3: './weights/B/data3_79.60_epoch128.pth'
        }
        # Original number of classes for each teacher model in Task B
        num_classes = {1: 5, 2: 5, 3: 5}
    
    for teacher_id in [1, 2, 3]:
        # Initialize model with the original number of classes
        model = CNN(num_classes=num_classes[teacher_id])
        model_path = model_paths[teacher_id]
        
        if os.path.exists(model_path):
            print(f"Loading teacher model {teacher_id} from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Warning: Teacher model {teacher_id} checkpoint not found at {model_path}")
            print(f"Initialize teacher model {teacher_id} with random weights")
            
        model.to(device)
        model.eval()  # Set to evaluation mode
        teacher_models[teacher_id] = model
    
    return teacher_models


def get_teacher_soft_labels(images, teacher_ids_batch, teacher_models, device, num_classes, temperature=2.0):
    """
    根据教师ID，获取教师模型的软标签
    对于由多个教师负责的类别，将相关教师输出的概率分布进行平均
    """
    batch_size = images.size(0)
    
    # 创建张量存储批次的软标签
    batch_soft_labels = torch.zeros(batch_size, num_classes).to(device)
    
    # 处理批次中的每个样本
    for i in range(batch_size):
        image = images[i].unsqueeze(0)  # 添加批次维度
        teacher_ids = teacher_ids_batch[i]
        
        # 跳过无教师分配的样本（理论上不应该发生）
        if not teacher_ids:
            continue
            
        # 创建张量记录每个教师对每个类别的预测次数
        teacher_counts = torch.zeros(num_classes).to(device)
        
        # 从每个相关教师获取预测
        for teacher_id in teacher_ids:
            with torch.no_grad():
                # 获取教师模型的输出
                teacher_output = teacher_models[teacher_id](image)
                
                # 应用温度缩放并转换为概率
                soft_probs = F.softmax(teacher_output / temperature, dim=1).squeeze()
                
                # 根据teacher_id将教师模型的输出映射到对应的全局类别
                # 使用从check_mappings.py中确认的映射关系
                
                # 将教师的预测添加到相应的全局类别位置
                if teacher_id == 1:
                    # 教师1本地索引0-4映射到全局索引
                    global_indices = [1, 7, 8, 9, 10]  # For teacher_1
                elif teacher_id == 2:
                    # 教师2本地索引0-4映射到全局索引
                    global_indices = [0, 1, 2, 6, 11]  # For teacher_2
                elif teacher_id == 3:
                    # 教师3本地索引0-4映射到全局索引
                    global_indices = [3, 4, 5, 9, 11]  # For teacher_3
                
                # 将教师的输出映射到全局类别空间
                for local_idx, global_idx in enumerate(global_indices):
                    batch_soft_labels[i, global_idx] += soft_probs[local_idx]
                    teacher_counts[global_idx] += 1
        
        # 对于被多个教师预测的类别，计算平均值
        valid_counts = teacher_counts > 0
        batch_soft_labels[i, valid_counts] /= teacher_counts[valid_counts]
    
    return batch_soft_labels


def distillation_loss(student_logits, teacher_soft_labels, labels, temperature=2.0, alpha=0.5):
    """
    计算知识蒸馏损失，结合：
    1. 学生预测与真实标签之间的交叉熵损失（硬标签）
    2. 学生与教师预测之间的KL散度（软标签）
    """
    # 硬标签损失
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # 软标签损失，使用温度缩放
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    # 教师软标签已经是概率分布
    soft_loss = F.kl_div(soft_student, teacher_soft_labels, reduction='batchmean') * (temperature ** 2)
    
    # 组合损失
    return (1 - alpha) * hard_loss + alpha * soft_loss


def train_step(student_model, teacher_models, images, labels, teacher_ids, optimizer, device, temperature=2.0, alpha=0.5):
    """
    执行单次知识蒸馏训练步骤
    """
    # 将数据移至设备
    images, labels = images.to(device), labels.to(device)
    
    # 学生模型前向传播
    student_logits = student_model(images)
    
    # 根据教师ID获取教师软标签
    teacher_soft_labels = get_teacher_soft_labels(images, teacher_ids, teacher_models, device, student_logits.size(1), temperature)
    
    # 计算蒸馏损失
    loss = distillation_loss(student_logits, teacher_soft_labels, labels, temperature, alpha)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 计算准确率
    _, predictions = torch.max(student_logits, 1)
    correct = (predictions == labels).sum().item()
    accuracy = correct / images.size(0)
    
    return loss.item(), accuracy


def validate(student_model, dataloader, device):
    """
    Validate the student model on the validation dataset
    """
    student_model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = student_model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    student_model.train()
    return avg_loss, accuracy


def plot_learning_curve(num_epochs, train_losses, train_accuracies, val_accuracies):
    """
    Plot the learning curves for loss and accuracy
    """
    # Smoothing the curves
    if len(train_losses) > 10:
        train_losses_smooth = savgol_filter(train_losses, min(15, len(train_losses) - 2 - (len(train_losses) - 2) % 2), 2)
        train_accuracies_smooth = savgol_filter(train_accuracies, min(15, len(train_accuracies) - 2 - (len(train_accuracies) - 2) % 2), 2)
        val_accuracies_smooth = savgol_filter(val_accuracies, min(15, len(val_accuracies) - 2 - (len(val_accuracies) - 2) % 2), 2)
    else:
        train_losses_smooth = train_losses
        train_accuracies_smooth = train_accuracies
        val_accuracies_smooth = val_accuracies

    epochs = range(1, num_epochs + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', alpha=0.3, label='Training Loss (raw)')
    plt.plot(epochs, train_losses_smooth, 'b-', label='Training Loss (smoothed)')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', alpha=0.3, label='Training Accuracy (raw)')
    plt.plot(epochs, train_accuracies_smooth, 'b-', label='Training Accuracy (smoothed)')
    plt.plot(epochs, val_accuracies, 'r-', alpha=0.3, label='Validation Accuracy (raw)')
    plt.plot(epochs, val_accuracies_smooth, 'r-', label='Validation Accuracy (smoothed)')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./learning_curve_kd.png')
    plt.close()


def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Task selection (A or B)
    task = 'B'  # Change to 'A' for Task A
    
    # Command line argument for task
    if len(sys.argv) > 1 and sys.argv[1] in ['A', 'B']:
        task = sys.argv[1]
    
    print(f"Running Task {task}")
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-3
    weight_decay = 5e-4
    temperature = 2.0  # Temperature for knowledge distillation
    alpha = 0.1  # Balance between hard and soft targets
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets with teacher IDs
    train_loader, val_loader, num_classes = load_all_datasets(batch_size=batch_size, task=task)
    print(f"Number of classes: {num_classes}")
    
    # Initialize student model
    student_model = StudentModel(num_classes=num_classes)
    student_model.to(device)
    
    # Load pre-trained teacher models with their original class counts
    teacher_models = load_teacher_models(device, task=task)
    
    # Optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Output directory based on task
    output_dir = f'./checkpoints/task_{task}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        # Training
        student_model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        batch_count = 0
        
        for images, labels, teacher_ids in train_loader:
            loss, accuracy = train_step(
                student_model, teacher_models, images, labels, teacher_ids, 
                optimizer, device, temperature, alpha
            )
            epoch_loss += loss
            epoch_accuracy += accuracy
            batch_count += 1
        
        # Average loss and accuracy for the epoch
        epoch_loss /= batch_count
        epoch_accuracy *= 100.0  # Convert to percentage
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Validation
        val_loss, val_accuracy = validate(student_model, val_loader, device)
        val_accuracies.append(val_accuracy)
        
        # Learning rate adjustment
        scheduler.step()
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(student_model.state_dict(), f'{output_dir}/student_model_best.pth')
            print(f"Saved best model with validation accuracy: {best_val_accuracy:.2f}%")
    
    # Save final model
    torch.save(student_model.state_dict(), f'{output_dir}/student_model_final.pth')
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-', alpha=0.3, label='Training Loss (raw)')
    plt.plot(range(1, num_epochs + 1), savgol_filter(train_losses, min(15, len(train_losses) - 2 - (len(train_losses) - 2) % 2), 2), 'b-', label='Training Loss (smoothed)')
    plt.title(f'Task {task} - Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, 'b-', alpha=0.3, label='Training Accuracy (raw)')
    plt.plot(range(1, num_epochs + 1), savgol_filter(train_accuracies, min(15, len(train_accuracies) - 2 - (len(train_accuracies) - 2) % 2), 2), 'b-', label='Training Accuracy (smoothed)')
    plt.plot(range(1, num_epochs + 1), val_accuracies, 'r-', alpha=0.3, label='Validation Accuracy (raw)')
    plt.plot(range(1, num_epochs + 1), savgol_filter(val_accuracies, min(15, len(val_accuracies) - 2 - (len(val_accuracies) - 2) % 2), 2), 'r-', label='Validation Accuracy (smoothed)')
    plt.title(f'Task {task} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curve_kd.png')
    plt.close()
    
    print(f"Training completed for Task {task}!")


if __name__ == "__main__":
    # Create checkpoints directory if it doesn't exist
    os.makedirs('./checkpoints', exist_ok=True)
    main()