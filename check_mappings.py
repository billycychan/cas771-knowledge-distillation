import torch
import os
import numpy as np
from collections import defaultdict

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

def main():
    # Path to each teacher model's training data
    # Task A
    task = 'A'
    if task == 'A':
        data_paths = {
            1: './data/TaskA/Model1_trees_superclass/model1_train_supercls.pth', 
            2: './data/TaskA/Model2_flowers_superclass/model2_train_supercls.pth', 
            3: './data/TaskA/Model3_fruit+veg_superclass/model3_train_supercls.pth'
        }
    
    # Task B
    elif task == 'B':
        data_paths = {
            1: './data/TaskB/train_dataB_model_1.pth',
            2: './data/TaskB/train_dataB_model_2.pth',
            3: './data/TaskB/train_dataB_model_3.pth'
        }
    
    # Dictionary to store the original class labels for each teacher model
    teacher_original_classes = {}
    # Dictionary to store the remapped class indices (0-4) for each teacher model
    teacher_local_mappings = {}
    # Dictionary to collect all unique classes across all models
    all_unique_classes = set()
    # Map each class to the teacher model(s) that trained on it
    class_to_teachers = defaultdict(list)
    
    # First, collect all unique classes and their mappings
    for teacher_id in [1, 2, 3]:
        data_path = data_paths[teacher_id]
        print(f"Loading data for Teacher {teacher_id} from {data_path}")
        
        # Check if file exists
        if not os.path.exists(data_path):
            print(f"Warning: File {data_path} does not exist!")
            continue
        
        if task == 'A':
            _, labels, _ = load_data(data_path, task)
        else:
            _, labels = load_data(data_path, task)
        
        # Get unique original class labels for this teacher
        unique_labels = sorted(set(labels))
        teacher_original_classes[teacher_id] = unique_labels
        all_unique_classes.update(unique_labels)
        
        # Create local mapping (0-4) for this teacher
        local_mapping = {label: i for i, label in enumerate(unique_labels)}
        teacher_local_mappings[teacher_id] = local_mapping
        
        # Record which teachers are responsible for which classes
        for label in unique_labels:
            class_to_teachers[label].append(teacher_id)
    
    # Create global mapping for all unique classes
    all_unique_classes = sorted(list(all_unique_classes))
    global_mapping = {label: i for i, label in enumerate(all_unique_classes)}
    
    # Print results
    print("\n=== Teacher Models Original Classes ===")
    for teacher_id, classes in teacher_original_classes.items():
        print(f"Teacher {teacher_id} classes: {classes}")
    
    print("\n=== Local Class Mappings (within each teacher model) ===")
    for teacher_id, mapping in teacher_local_mappings.items():
        print(f"Teacher {teacher_id} local mapping:")
        for orig_class, local_idx in mapping.items():
            print(f"  Original class {orig_class} → local index {local_idx}")
    
    print("\n=== Global Mapping (for Knowledge Distillation) ===")
    for orig_class, global_idx in global_mapping.items():
        teacher_ids = class_to_teachers[orig_class]
        print(f"Original class {orig_class} → global index {global_idx}, taught by teachers: {teacher_ids}")
    
    print("\n=== Mapping to Use in train.py ===")
    for teacher_id in [1, 2, 3]:
        if teacher_id not in teacher_local_mappings:
            continue
            
        local_to_global = {}
        for orig_class, local_idx in teacher_local_mappings[teacher_id].items():
            global_idx = global_mapping[orig_class]
            local_to_global[local_idx] = global_idx
        
        print(f"Teacher {teacher_id} mapping for train.py:")
        indices = [local_to_global[i] for i in range(len(teacher_local_mappings[teacher_id]))]
        print(f"global_indices = {indices}  # For teacher_{teacher_id}")

if __name__ == "__main__":
    main()