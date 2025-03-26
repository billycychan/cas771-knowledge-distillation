import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np

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

def load_data(data_path, task='A'):
    raw_data = torch.load(data_path)
    if task == 'A':
        data = raw_data['data']
        labels = raw_data['labels']
        indices = raw_data['indices']
        return data, labels, indices
    elif task == 'B':
        data = raw_data['data'].numpy().transpose(0, 3, 1, 2)
        labels = raw_data['labels'].numpy()
    return data, labels, None

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

def verify_dataset(data, labels, dataset_id, split_name):
    """Verify dataset integrity"""
    print(f"\n--- Verifying Dataset {dataset_id} ({split_name}) ---")
    
    # Basic shape information
    if isinstance(data, torch.Tensor):
        shape_str = "x".join([str(dim) for dim in data.shape])
        dtype_str = str(data.dtype)
    else:
        shape_str = "x".join([str(dim) for dim in data.shape])
        dtype_str = str(type(data))
    
    print(f"Data shape: {shape_str}, dtype: {dtype_str}")
    
    # Label information
    unique_labels = sorted(set(labels))
    # label_counts = {label: labels.count(label) for label in unique_labels}
    label_counts = {label: np.count_nonzero(labels == label) for label in unique_labels}
    print(f"Labels: {len(labels)} samples, {len(unique_labels)} classes")
    print(f"Class distribution: {label_counts}")
    
    # Basic data statistics
    if isinstance(data, torch.Tensor):
        data_flat = data.view(data.size(0), -1)
        data_min = data_flat.min().item()
        data_max = data_flat.max().item()
        data_mean = data_flat.mean().item()
        data_std = data_flat.std().item()
    else:
        data_flat = data.reshape(data.shape[0], -1)
        data_min = data_flat.min()
        data_max = data_flat.max()
        data_mean = data_flat.mean()
        data_std = data_flat.std()
    
    print(f"Data range: [{data_min:.4f}, {data_max:.4f}], mean: {data_mean:.4f}, std: {data_std:.4f}")
    print(f"--- Verification completed for Dataset {dataset_id} ---")
    
    return True

def load_all_datasets(batch_size=32, task='A'):
    """Load and combine all three datasets for 15-class classification"""
    print(f"\n====== Loading All Datasets for Task {task} ======")
    # Load each dataset
    if task == 'A':
        data_paths = {
            1: ('./data/TaskA/Model1_trees_superclass/model1_train_supercls.pth', 
                './data/TaskA/Model1_trees_superclass/model1_test_supercls.pth'),
            2: ('./data/TaskA/Model2_flowers_superclass/model2_train_supercls.pth', 
                './data/TaskA/Model2_flowers_superclass/model2_test_supercls.pth'),
            3: ('./data/TaskA/Model3_fruit+veg_superclass/model3_train_supercls.pth', 
                './data/TaskA/Model3_fruit+veg_superclass/model3_test_supercls.pth')
        }
    else:  # task B
        data_paths = {
            1: ('./data/TaskB/train_dataB_model_1.pth', 
                './data/TaskB/val_dataB_model_1.pth'),
            2: ('./data/TaskB/train_dataB_model_2.pth', 
                './data/TaskB/val_dataB_model_2.pth'),
            3: ('./data/TaskB/train_dataB_model_3.pth', 
                './data/TaskB/val_dataB_model_3.pth')
        }
        
    all_train_data = []
    all_train_labels = []
    all_test_data = []
    all_test_labels = []
    
    class_offset = 0
    total_classes = 0
    
    # First pass: determine the total number of classes and create mappings
    for model_id in [1, 2, 3]:
        train_data_path, _ = data_paths[model_id]
        _, train_labels, _ = load_data(train_data_path, task)
        unique_labels = sorted(set(train_labels))
        total_classes += len(unique_labels)
        print(f"Dataset {model_id} has {len(unique_labels)} classes: {unique_labels}")
    
    print(f"Total classes across all datasets: {total_classes}")
    
    # Second pass: load and remap labels
    class_offset = 0
    for model_id in [1, 2, 3]:
        train_data_path, test_data_path = data_paths[model_id]
        
        # Load data
        train_data, train_labels, _ = load_data(train_data_path, task)
        test_data, test_labels, _ = load_data(test_data_path, task)
        
        # Verify individual datasets before mapping
        verify_dataset(train_data, train_labels, model_id, "train")
        verify_dataset(test_data, test_labels, model_id, "test")
        
        # Unique labels in this dataset
        unique_labels = sorted(set(train_labels))
        num_classes = len(unique_labels)
        
        # Create mapping dictionary
        label_map = {orig_label: new_label + class_offset for new_label, orig_label in enumerate(unique_labels)}
        print(f"Dataset {model_id} label mapping: {label_map}")
        
        # Apply mapping
        mapped_train_labels = [label_map[label] for label in train_labels]
        mapped_test_labels = [label_map[label] for label in test_labels]
        
        # Verify label range
        min_train = min(mapped_train_labels)
        max_train = max(mapped_train_labels)
        min_test = min(mapped_test_labels)
        max_test = max(mapped_test_labels)
        print(f"Dataset {model_id} remapped label range - Train: {min_train}-{max_train}, Test: {min_test}-{max_test}")
        
        # Update the class offset for the next dataset
        class_offset += num_classes
        
        # Add to combined dataset
        all_train_data.append(train_data)
        all_train_labels.extend(mapped_train_labels)
        all_test_data.append(test_data)
        all_test_labels.extend(mapped_test_labels)
    
    # Combine data
    print(f"Data type check: {type(all_train_data[0])}")
    if not isinstance(all_train_data[0], torch.Tensor):
        all_train_data = [torch.from_numpy(arr) for arr in all_train_data]
        all_test_data = [torch.from_numpy(arr) for arr in all_test_data]
    all_train_data = torch.cat(all_train_data, dim=0)
    all_test_data = torch.cat(all_test_data, dim=0)
    
    # Convert labels to tensors
    all_train_labels = torch.tensor(all_train_labels, dtype=torch.long)
    all_test_labels = torch.tensor(all_test_labels, dtype=torch.long)
    
    # Final verification
    min_label = all_train_labels.min().item()
    max_label = all_train_labels.max().item()
    print(f"Final label range: {min_label} to {max_label} (expected 0 to {class_offset-1})")
    
    if min_label < 0 or max_label >= class_offset:
        raise ValueError(f"Labels out of expected range! Got {min_label} to {max_label}, expected 0 to {class_offset-1}")
    
    # Verify combined dataset
    print("\n--- Verifying Combined Dataset ---")
    print(f"Combined training data: {all_train_data.shape}, labels: {all_train_labels.shape}")
    print(f"Combined test data: {all_test_data.shape}, labels: {all_test_labels.shape}")
    
    # Display label distribution
    train_label_counts = {}
    for i in range(total_classes):
        train_label_counts[i] = (all_train_labels == i).sum().item()
    
    test_label_counts = {}
    for i in range(total_classes):
        test_label_counts[i] = (all_test_labels == i).sum().item()
    
    print(f"Training set class distribution: {train_label_counts}")
    print(f"Test set class distribution: {test_label_counts}")
    
    # Calculate normalization statistics
    initial_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = CAS771Dataset(all_train_data, all_train_labels, transform=initial_transform)
    train_loader_for_stats = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    mean, std = calculate_normalization_stats(train_loader_for_stats)
    print(f"Dataset mean: {mean}, std: {std}")
    
    # Apply normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(size=all_train_data.shape[2:], scale=(0.8, 1.0))
    ])
    
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    # ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])
    
    # Create final datasets and dataloaders
    train_dataset = CAS771Dataset(all_train_data, all_train_labels, transform=transform)
    test_dataset = CAS771Dataset(all_test_data, all_test_labels, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Final train loader: {len(train_loader)} batches, batch size: {batch_size}")
    print(f"Final test loader: {len(test_loader)} batches, batch size: {batch_size}")
    print("====== Data Loading Complete ======\n")
    
    return train_loader, test_loader, total_classes 