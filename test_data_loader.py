import torch
import os
from teacher_model_A import load_and_prepare_data as load_data_A
from teacher_model_B import load_and_prepare_data as load_data_B
from merge_data_loader import load_all_datasets as merge_load_all_datasets
from data_loader_deprecated import load_all_datasets as data_loader_load_all_datasets

def test_single_model_loader(loader_func, model_id, dataset_name):
    print(f"\nTesting {dataset_name} Dataset Model {model_id} Loader:")
    try:
        train_loader, test_loader, num_classes = loader_func(model_used=model_id, batch_size=32)
        
        # Test train loader
        sample_batch = next(iter(train_loader))
        images, labels = sample_batch
        print(f"Train Loader - Batch shape: {images.shape}, Labels shape: {labels.shape}")
        print(f"Label range: {labels.min().item()} to {labels.max().item()}")
        print(f"Number of classes: {num_classes}")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of test batches: {len(test_loader)}")
        print("✓ Loader test passed successfully!")
        return True
    except Exception as e:
        print(f"✗ Error testing loader: {str(e)}")
        return False

def test_merged_loader(loader_func, task, loader_name):
    print(f"\nTesting {loader_name} for Task {task}:")
    try:
        train_loader, test_loader, total_classes = loader_func(batch_size=32, task=task)
        
        # Test train loader
        sample_batch = next(iter(train_loader))
        images, labels = sample_batch
        print(f"Train Loader - Batch shape: {images.shape}, Labels shape: {labels.shape}")
        print(f"Label range: {labels.min().item()} to {labels.max().item()}")
        print(f"Total number of classes: {total_classes}")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of test batches: {len(test_loader)}")
        print("✓ Merged loader test passed successfully!")
        return True
    except Exception as e:
        print(f"✗ Error testing merged loader: {str(e)}")
        return False

def main():
    print("Starting Data Loader Tests...")
    
    # Test Model A loader
    for model_id in [1, 2, 3]:
        test_single_model_loader(load_data_A, model_id, "A")
    
    # Test Model B loader
    for model_id in [1, 2, 3]:
        test_single_model_loader(load_data_B, model_id, "B")
    
    # Test merged loaders from merge_data_loader.py
    test_merged_loader(merge_load_all_datasets, 'A', "Merged Dataset Loader")
    test_merged_loader(merge_load_all_datasets, 'B', "Merged Dataset Loader")
    
    # ! deprecated
    # # Test data_loader.py loader
    # test_merged_loader(data_loader_load_all_datasets, 'A', "Data Loader")
    # test_merged_loader(data_loader_load_all_datasets, 'B', "Data Loader")

if __name__ == "__main__":
    main()
