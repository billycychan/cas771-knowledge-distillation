import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data_loader_kd import load_all_datasets
import time

def test_dataloader(task='B'):
    print(f"\n=== Testing DataLoader for Task {task} ===")
    
    # Load datasets
    print("Loading datasets...")
    start_time = time.time()
    train_loader, test_loader, total_classes = load_all_datasets(batch_size=32, task=task)
    load_time = time.time() - start_time
    print(f"Datasets loaded in {load_time:.2f} seconds")
    
    # Print dataset info
    print(f"Total classes: {total_classes}")
    
    # Count samples in train and test loaders
    train_samples = 0
    train_class_counts = {}
    train_teacher_id_counts = {}
    shared_classes = 0
    
    print("\nProcessing training dataset...")
    batch_sizes = []
    for batch_idx, (images, labels, teacher_ids) in enumerate(train_loader):
        batch_sizes.append(images.shape[0])
        batch_samples = images.shape[0]
        train_samples += batch_samples
        
        # Process labels
        for label in labels:
            label_item = label.item()
            if label_item not in train_class_counts:
                train_class_counts[label_item] = 0
            train_class_counts[label_item] += 1
        
        # Process teacher IDs
        for teacher_id in teacher_ids:
            # Convert to tuple for hashability
            teacher_id_tuple = tuple(sorted(teacher_id))
            if teacher_id_tuple not in train_teacher_id_counts:
                train_teacher_id_counts[teacher_id_tuple] = 0
            train_teacher_id_counts[teacher_id_tuple] += 1
            
            if len(teacher_id) > 1:
                shared_classes += 1
        
        # Print info for first few batches
        if batch_idx < 2:
            print(f"Batch {batch_idx+1} - Shape: {images.shape}, Labels shape: {labels.shape}")
            print(f"  Sample labels: {labels[:5].tolist()}")
            print(f"  Sample teacher_ids: {[tid for tid in teacher_ids[:5]]}")
    
    # Count test samples
    test_samples = sum(len(batch[0]) for batch in test_loader)
    
    print("\n=== Dataset Statistics ===")
    print(f"Training samples: {train_samples}")
    print(f"Test samples: {test_samples}")
    print(f"Average batch size: {np.mean(batch_sizes):.2f}")
    
    print("\n=== Class Distribution ===")
    for class_id, count in sorted(train_class_counts.items()):
        print(f"Class {class_id}: {count} samples")
    
    print("\n=== Teacher ID Distribution ===")
    for teacher_id, count in sorted(train_teacher_id_counts.items()):
        print(f"Teacher ID {teacher_id}: {count} samples")
    
    print(f"\nShared classes: {shared_classes} samples")
    print(f"Single teacher classes: {train_samples - shared_classes} samples")
    
    # Verify a complete epoch by iterating through the entire dataloader
    print("\nVerifying complete epoch traversal...")
    complete_samples = 0
    start_time = time.time()
    for batch_idx, (images, labels, _) in enumerate(train_loader):
        complete_samples += images.shape[0]
        if batch_idx % 10 == 0 and batch_idx > 0:
            print(f"Processed {batch_idx} batches, {complete_samples} samples")
    
    epoch_time = time.time() - start_time
    print(f"Complete epoch ({complete_samples} samples) processed in {epoch_time:.2f} seconds")
    assert complete_samples == train_samples, f"Mismatch in sample count: {complete_samples} vs expected {train_samples}"
    
    print("\nDataLoader test completed successfully!")
    return train_loader, test_loader

def visualize_samples(loader, num_samples=5):
    """Visualize some samples from the dataset"""
    # Get a batch
    images, labels, teacher_ids = next(iter(loader))
    
    # Create a figure
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    fig.suptitle("Sample Images from Dataset")
    
    # Plot each image
    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).numpy()
        # Denormalize
        img = img * np.array([0.2402, 0.2305, 0.2377]) + np.array([0.5089, 0.5356, 0.6049])
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {labels[i].item()}\nTeacher: {teacher_ids[i]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    print("Saved sample images to 'sample_images.png'")

if __name__ == "__main__":
    print("===== DataLoader_KD Test Script =====")
    
    # Test for Task B
    train_loader, test_loader = test_dataloader(task='B')
    
    # Uncomment to test Task A as well
    # test_dataloader(task='A')
    
    # Visualize some samples
    try:
        visualize_samples(train_loader)
    except Exception as e:
        print(f"Visualization error: {e}")
