import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

def custom_collate_fn(batch):
    """Custom collate function to handle teacher_ids that may be lists of different lengths."""
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    teacher_ids = [item[2] for item in batch]  # Keep as list of lists
    return images, labels, teacher_ids

class CAS771Dataset(Dataset):
    def __init__(self, data, labels, teacher_ids, transform=None):
        self.data = data
        self.labels = labels
        self.teacher_ids = teacher_ids
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        teacher_id = self.teacher_ids[idx]
        if self.transform:
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            else:
                img = transforms.ToPILImage()(img)
            img = self.transform(img)
        return img, label, teacher_id

def load_data(data_path, task='B'):
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
    for images, _, teacher_ids in dataloader:
        channel_sum += torch.mean(images, dim=[0,2,3]) * images.size(0)
        channel_sum_sq += torch.mean(images ** 2, dim=[0,2,3]) * images.size(0)
        num_pixels += images.size(0)

    # Calculate mean and std
    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sum_sq / num_pixels - mean ** 2)

    return mean, std

def load_all_datasets(batch_size=32, task='B'):
    """Load and combine all three datasets for 15-class classification"""
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
        
    # First pass: collect all unique labels across all datasets and identify shared classes
    all_unique_labels = set()
    dataset_labels = {}  # Store labels for each dataset
    label_to_teacher_ids = {}  # Map each label to the list of teacher models it appears in
    
    for model_id in [1, 2, 3]:
        train_data_path, _ = data_paths[model_id]
        _, train_labels, _ = load_data(train_data_path, task)
        unique_labels = sorted(set(train_labels))
        dataset_labels[model_id] = unique_labels
        all_unique_labels.update(unique_labels)
        
        # Add this teacher model to the list of teachers for each label
        for label in unique_labels:
            if label not in label_to_teacher_ids:
                label_to_teacher_ids[label] = []
            label_to_teacher_ids[label].append(model_id)
    
    # Create a global mapping for all unique labels
    all_unique_labels = sorted(list(all_unique_labels))
    global_label_map = {orig_label: new_label for new_label, orig_label in enumerate(all_unique_labels)}
    total_classes = len(all_unique_labels)
    
    print(f"Total unique classes across all datasets: {total_classes}")
    print(f"Global label mapping: {global_label_map}")
    
    # Print shared classes
    shared_classes = [label for label, teachers in label_to_teacher_ids.items() if len(teachers) > 1]
    print(f"Shared classes across teacher models: {shared_classes}")
    for label in shared_classes:
        print(f"Class {label} (mapped to {global_label_map[label]}) is shared by teachers: {label_to_teacher_ids[label]}")
    
    # Store data per class
    class_data = {global_label_map[label]: {'train_data': [], 'train_labels': [], 'test_data': [], 'test_labels': [], 
                                          'teacher_ids': label_to_teacher_ids[label]} 
                 for label in all_unique_labels}
    
    # Second pass: load data and organize by class
    for model_id in [1, 2, 3]:
        train_data_path, test_data_path = data_paths[model_id]
        
        # Load data
        train_data, train_labels, _ = load_data(train_data_path, task)
        test_data, test_labels, _ = load_data(test_data_path, task)
        
        # For shared classes, we only want to include samples from the first teacher model
        # For non-shared classes, include all samples from their respective models
        
        # Add training data
        for i, label in enumerate(train_labels):
            mapped_label = global_label_map[label]
            
            # For shared classes, only add data from the first teacher model
            if label in shared_classes and model_id != min(label_to_teacher_ids[label]):
                continue
                
            class_data[mapped_label]['train_data'].append(train_data[i])
            class_data[mapped_label]['train_labels'].append(mapped_label)
        
        # Add test data
        for i, label in enumerate(test_labels):
            mapped_label = global_label_map[label]
            
            # For shared classes, only add data from the first teacher model
            if label in shared_classes and model_id != min(label_to_teacher_ids[label]):
                continue
                
            class_data[mapped_label]['test_data'].append(test_data[i])
            class_data[mapped_label]['test_labels'].append(mapped_label)
    
    # Combine data
    all_train_data = []
    all_train_labels = []
    all_train_teacher_ids = []
    all_test_data = []
    all_test_labels = []
    all_test_teacher_ids = []
    
    for mapped_label, data in class_data.items():
        all_train_data.extend(data['train_data'])
        all_train_labels.extend(data['train_labels'])
        all_train_teacher_ids.extend([data['teacher_ids']] * len(data['train_data']))
        
        all_test_data.extend(data['test_data'])
        all_test_labels.extend(data['test_labels'])
        all_test_teacher_ids.extend([data['teacher_ids']] * len(data['test_data']))
    
    # Convert to tensor format
    if not isinstance(all_train_data[0], torch.Tensor):
        all_train_data = [torch.from_numpy(arr) for arr in all_train_data]
        all_test_data = [torch.from_numpy(arr) for arr in all_test_data]
    
    all_train_data = torch.stack(all_train_data)
    all_test_data = torch.stack(all_test_data)
    
    # Convert labels to tensors
    all_train_labels = torch.tensor(all_train_labels, dtype=torch.long)
    all_test_labels = torch.tensor(all_test_labels, dtype=torch.long)
    
    # Note: teacher_ids are now lists, so we keep them as Python lists
    
    # Final verification
    min_label = all_train_labels.min().item()
    max_label = all_train_labels.max().item()
    print(f"Final label range: {min_label} to {max_label} (expected 0 to {total_classes-1})")
    
    if min_label < 0 or max_label >= total_classes:
        raise ValueError(f"Labels out of expected range! Got {min_label} to {max_label}, expected 0 to {total_classes-1}")
    
    # Calculate normalization statistics
    initial_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = CAS771Dataset(all_train_data, all_train_labels, all_train_teacher_ids, transform=initial_transform)
    train_loader_for_stats = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                       num_workers=2, pin_memory=True, collate_fn=custom_collate_fn)
    
    mean, std = calculate_normalization_stats(train_loader_for_stats)
    print(f"Dataset mean: {mean}, std: {std}")
    
    # Apply normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])
    
    # Create final datasets and dataloaders
    train_dataset = CAS771Dataset(all_train_data, all_train_labels, all_train_teacher_ids, transform=transform)
    test_dataset = CAS771Dataset(all_test_data, all_test_labels, all_test_teacher_ids, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
    
    return train_loader, test_loader, total_classes