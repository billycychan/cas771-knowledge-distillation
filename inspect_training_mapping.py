import torch
import os

def load_data(data_path, task='B'):
    raw_data = torch.load(data_path)
    if task == 'A':
        data = raw_data['data']
        labels = raw_data['labels']
        indices = raw_data['indices']
        return data, labels, indices
    elif task == 'B':
        data = raw_data['data']
        labels = raw_data['labels']
    return data, labels, None

def main():
    # Define paths to training data for each model subset
    data_paths = {
        1: './data/TaskB/train_dataB_model_1.pth',
        2: './data/TaskB/train_dataB_model_2.pth',
        3: './data/TaskB/train_dataB_model_3.pth'
    }
    
    # First pass: collect all unique labels across all datasets
    all_unique_labels = set()
    dataset_labels = {}  # Store labels for each dataset
    label_to_teacher_ids = {}  # Map each label to the list of teacher models it appears in
    
    for model_id in [1, 2, 3]:
        train_data_path = data_paths[model_id]
        _, train_labels, _ = load_data(train_data_path, 'B')
        
        # Convert tensor labels to integers if needed
        if isinstance(train_labels[0], torch.Tensor):
            train_labels = [label.item() for label in train_labels]
            
        unique_labels = sorted(set(train_labels))
        dataset_labels[model_id] = unique_labels
        all_unique_labels.update(unique_labels)
        
        # Add this teacher model to the list of teachers for each label
        for label in unique_labels:
            if label not in label_to_teacher_ids:
                label_to_teacher_ids[label] = []
            label_to_teacher_ids[label].append(model_id)
    
    # Create a global mapping for all unique labels - SAME AS TRAINING CODE
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
    
    # Print class names for reference
    class_names = {
        24: "African_elephant",
        34: "hyena",
        80: "zebra",
        124: "collie",
        125: "golden_retriever",
        130: "boxer",
        135: "patas",
        137: "baboon",
        159: "Arctic_fox",
        173: "Chihuahua",
        201: "lynx",
        202: "African_hunting_dog"
    }
    
    print("\nClass mapping with names:")
    for orig_label, new_label in global_label_map.items():
        class_name = class_names.get(orig_label, f"Unknown-{orig_label}")
        print(f"Original label {orig_label} ({class_name}) → Global index {new_label}")
    
    # Now check validation data
    print("\n\nValidation Data:")
    val_paths = {
        1: './data/TaskB/val_dataB_model_1.pth',
        2: './data/TaskB/val_dataB_model_2.pth',
        3: './data/TaskB/val_dataB_model_3.pth'
    }
    
    # Check validation data distribution
    val_label_counts = {}
    for model_id, path in val_paths.items():
        print(f"Loading validation data from {path}")
        _, val_labels, _ = load_data(path, 'B')
        
        # Convert tensor labels to integers if needed
        if isinstance(val_labels[0], torch.Tensor):
            val_labels = [label.item() for label in val_labels]
            
        # Count labels
        for label in val_labels:
            if label not in val_label_counts:
                val_label_counts[label] = 0
            val_label_counts[label] += 1
    
    print("\nValidation data distribution:")
    for label, count in sorted(val_label_counts.items()):
        class_name = class_names.get(label, f"Unknown-{label}")
        global_idx = global_label_map.get(label, "Not in mapping")
        print(f"Original label {label} ({class_name}) → Global index {global_idx}: {count} samples")

if __name__ == "__main__":
    main()
