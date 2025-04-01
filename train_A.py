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
import datetime
import time
import argparse

# Import custom modules
from data_loader_kd import load_all_datasets, custom_collate_fn
from teacher_model_B import CNN, set_seed
from utils import plot_learning_curve
from student_model import StudentModel  # Import the Swin Transformer student model

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
    Get soft labels from teacher models based on teacher IDs.
    For classes covered by multiple teachers, average their probability distributions.
    """
    batch_size = images.size(0)
    batch_soft_labels = torch.zeros(batch_size, num_classes).to(device)
    
    # Create mapping dictionary - precompute instead of repeating in the loop
    teacher_mappings = {
        1: [1, 3, 6, 8, 14],  # Teacher 1 local indices 0-4 mapping to global indices
        2: [5, 9, 10, 11, 13],  # Teacher 2 local indices 0-4 mapping to global indices
        3: [0, 2, 4, 7, 12]   # Teacher 3 local indices 0-4 mapping to global indices
    }
    
    # Process by teacher model group - more efficient batch processing
    for teacher_id in [1, 2, 3]:
        # Find indices of all samples this teacher is responsible for
        indices = [i for i, teacher_ids in enumerate(teacher_ids_batch) if teacher_id in teacher_ids]
        
        if not indices:
            continue  # No samples use this teacher, skip
            
        # Get all samples this teacher is responsible for, process at once
        teacher_batch = images[indices]
        
        # Get teacher model predictions for the entire batch at once
        with torch.no_grad():
            teacher_outputs = teacher_models[teacher_id](teacher_batch)
            
            # Apply temperature scaling and convert to probabilities
            soft_probs = F.softmax(teacher_outputs / temperature, dim=1)
            
            # Get mapping relationship
            global_indices = teacher_mappings[teacher_id]
            
            # Record teacher predictions and counts for each sample
            for batch_idx, orig_idx in enumerate(indices):
                for local_idx, global_idx in enumerate(global_indices):
                    batch_soft_labels[orig_idx, global_idx] += soft_probs[batch_idx, local_idx]
    
    # For each sample, calculate how many times each class was predicted
    teacher_counts = torch.zeros(batch_size, num_classes).to(device)
    for i, teacher_ids in enumerate(teacher_ids_batch):
        for teacher_id in teacher_ids:
            for global_idx in teacher_mappings[teacher_id]:
                teacher_counts[i, global_idx] += 1
    
    # Calculate averages
    valid_mask = teacher_counts > 0
    batch_soft_labels[valid_mask] /= teacher_counts[valid_mask]
    
    return batch_soft_labels


def distillation_loss(student_logits, teacher_soft_labels, labels, temperature=2.0, alpha=0.5):
    """
    Calculate knowledge distillation loss, combining:
    1. Cross entropy loss between student predictions and true labels (hard labels)
    2. KL divergence between student and teacher predictions (soft labels)
    """
    # Hard label loss
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Soft label loss with temperature scaling
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    # Teacher soft labels are already probability distributions
    soft_loss = F.kl_div(soft_student, teacher_soft_labels, reduction='batchmean') * (temperature ** 2)
    
    # Combined loss
    return (1 - alpha) * hard_loss + alpha * soft_loss


def train_step(student_model, teacher_models, images, labels, teacher_ids, optimizer, device, temperature=2.0, alpha=0.5):
    """
    Execute a single knowledge distillation training step
    """
    # Move data to device
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass through student model
    student_logits = student_model(images)
    
    # Get teacher soft labels based on teacher IDs
    teacher_soft_labels = get_teacher_soft_labels(images, teacher_ids, teacher_models, device, student_logits.size(1), temperature)
    
    # Calculate distillation loss
    loss = distillation_loss(student_logits, teacher_soft_labels, labels, temperature, alpha)
    
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    _, predictions = torch.max(student_logits, 1)
    correct = (predictions == labels).sum().item()
    accuracy = 100 * correct / images.size(0)  # Convert to percentage to match validate function
    
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


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training with Multiple Teachers')
    parser.add_argument('--task', type=str, default='B', choices=['A', 'B'], help='Task to run (A or B)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for knowledge distillation')
    parser.add_argument('--alpha', type=float, default=0.1, help='Weight balance between hard and soft targets (higher means more weight on soft targets)')
    parser.add_argument('--scheduler_step', type=int, default=30, help='Step size for learning rate scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='Gamma for learning rate scheduler')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--student_model', type=str, default='EfficientNet', choices=['MobileNetV3', 'EfficientNet', 'ResNet18'], help='Student model architecture')
    parser.add_argument('--run', type=str, default='', help='Optional suffix to append to the log directory name')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create runs directory for logging
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'./runs/{current_time}_task_{args.task}'
    if args.run:
        log_dir += f'_{args.run}'
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging to file and console
    log_file = f'{log_dir}/training_log.txt'
    
    # Logger function to write to both console and file
    def log_print(*logs, **kwargs):
        print(*logs, **kwargs)
        with open(log_file, 'a') as f:
            print(*logs, file=f, **kwargs)
    
    log_print(f"Started training at {current_time}")
    log_print(f"Running Task {args.task}")
    
    # Log all arguments
    log_print("Training configuration:")
    for arg, value in sorted(vars(args).items()):
        log_print(f"- {arg}: {value}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_print(f"Using device: {device}")
    
    # Load datasets with teacher IDs
    train_loader, val_loader, num_classes = load_all_datasets(batch_size=args.batch_size, task=args.task)
    log_print(f"Number of classes: {num_classes}")
    
    # Import and select student model dynamically based on argument
    if args.student_model == 'MobileNetV3':
        from student_model import MobileNetV3Student as StudentModelClass
    elif args.student_model == 'EfficientNet':
        from student_model import EfficientNetStudent as StudentModelClass
    elif args.student_model == 'ResNet18':
        from student_model import ResNet18Student as StudentModelClass
    else:
        from student_model import StudentModel as StudentModelClass
    
    # Initialize student model
    student_model = StudentModelClass(num_classes=num_classes)
    student_model.to(device)
    log_print(f"Using student model: {args.student_model}")
    
    # Load pre-trained teacher models with their original class counts
    teacher_models = load_teacher_models(device, task=args.task)
    
    # Optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    
    # Training loop
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0
    prev_best_model_path = None  # Track the previous best model path
    
    start_time = time.time()
    log_print("\nStarting training...")
    
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        
        # Training
        student_model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        batch_count = 0
        
        for images, labels, teacher_ids in train_loader:
            loss, accuracy = train_step(
                student_model, teacher_models, images, labels, teacher_ids, 
                optimizer, device, args.temperature, args.alpha
            )
            epoch_loss += loss
            epoch_accuracy += accuracy
            batch_count += 1
        
        # Average loss and accuracy for the epoch
        epoch_loss /= batch_count
        epoch_accuracy /= batch_count
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Validation
        val_loss, val_accuracy = validate(student_model, val_loader, device)
        val_accuracies.append(val_accuracy)
        
        # Learning rate adjustment
        scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print progress
        log_print(f"Epoch [{epoch+1}/{args.num_epochs}] - "
                f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% "
                f"- Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            # Delete previous best model if it exists
            if prev_best_model_path is not None and os.path.exists(prev_best_model_path):
                os.remove(prev_best_model_path)
                log_print(f"Removed previous best model: {prev_best_model_path}")
            
            best_val_accuracy = val_accuracy
            # Save model in the runs directory with epoch and accuracy in the filename
            best_model_path = f'{log_dir}/student_model_epoch{epoch+1}_acc{val_accuracy:.2f}.pth'
            torch.save(student_model.state_dict(), best_model_path)
            prev_best_model_path = best_model_path  # Update the previous best model path
            log_print(f"Saved best model at epoch {epoch+1} with validation accuracy: {best_val_accuracy:.2f}%")
        
        # Save learning curve after each epoch (always overwrite the previous one)
        plot_learning_curve(
            epoch + 1, 
            train_losses, 
            train_accuracies, 
            val_accuracies, 
            save_path=f'{log_dir}/learning_curve.png'
        )
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    log_print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    log_print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    
    # Save final model
    final_model_path = f'{log_dir}/student_model_final_epoch{args.num_epochs}_acc{best_val_accuracy:.2f}.pth'
    torch.save(student_model.state_dict(), final_model_path)
    log_print(f"Final model saved to {final_model_path}")
    
    # Save final learning curve
    plot_learning_curve(
        args.num_epochs, 
        train_losses, 
        train_accuracies, 
        val_accuracies, 
        save_path=f'{log_dir}/learning_curve.png'
    )
    
    log_print(f"Training completed for Task {args.task}!")
    log_print(f"All logs and visualizations saved to {log_dir}")


if __name__ == "__main__":
    # Create checkpoints directory if it doesn't exist
    os.makedirs('./checkpoints', exist_ok=True)
    main()