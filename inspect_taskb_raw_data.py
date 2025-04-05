import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from data_loader_kd import load_data

def inspect_dataset_structure(val_file):
    """
    Inspect the structure of the dataset to understand its format
    """
    print(f"\nInspecting dataset: {val_file}")
    data, labels, _ = load_data(val_file, task='B')
    
    print(f"Data type: {type(data)}")
    print(f"Number of samples: {len(data)}")
    
    if len(data) > 0:
        sample = data[0]
        print(f"Sample type: {type(sample)}")
        
        if isinstance(sample, torch.Tensor):
            print(f"Sample shape: {sample.shape}")
            print(f"Sample data type: {sample.dtype}")
            print(f"Sample min value: {sample.min().item()}")
            print(f"Sample max value: {sample.max().item()}")
            print(f"Sample mean value: {sample.mean().item()}")
        elif isinstance(sample, np.ndarray):
            print(f"Sample shape: {sample.shape}")
            print(f"Sample data type: {sample.dtype}")
            print(f"Sample min value: {sample.min()}")
            print(f"Sample max value: {sample.max()}")
            print(f"Sample mean value: {sample.mean()}")
        else:
            print(f"Sample is of unexpected type")
    
    print(f"Labels type: {type(labels)}")
    print(f"Number of labels: {len(labels)}")
    
    if len(labels) > 0:
        print(f"First 10 labels: {labels[:10]}")
    
    return data, labels

def display_original_images_properly():
    """
    Display the original images from TaskB datasets with proper scaling and format handling
    """
    # Load data directly from each validation file
    val_files = [
        './data/TaskB/val_dataB_model_1.pth',  # Mammal dataset
        './data/TaskB/val_dataB_model_2.pth',  # African Animal dataset
        './data/TaskB/val_dataB_model_3.pth',  # Canidae dataset
    ]
    
    # Class mappings based on user-provided information
    class_names = {
        173: "Chihuahua",
        137: "baboon",
        34: "hyena", 
        159: "Arctic_fox", 
        201: "lynx",
        202: "African_hunting_dog", 
        80: "zebra", 
        135: "patas", 
        24: "African_elephant",
        130: "boxer", 
        124: "collie", 
        125: "golden_retriever"
    }
    
    dataset_names = ["Mammal", "African Animal", "Canidae"]
    
    # First inspect the datasets to understand their structure
    for val_file in val_files:
        data, _ = inspect_dataset_structure(val_file)
    
    # Now display images properly based on the dataset structure
    for dataset_idx, val_file in enumerate(val_files):
        print(f"\nDisplaying images from {val_file}")
        data, labels = load_data(val_file, task='B')
        
        # Randomly select 10 images to display
        indices = np.random.choice(range(len(data)), size=min(10, len(data)), replace=False)
        
        # Create a figure to display images
        rows = 2
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
        fig.suptitle(f"Original images from {dataset_names[dataset_idx]} dataset", fontsize=16)
        
        # Display each selected image
        for i, idx in enumerate(indices):
            row = i // cols
            col = i % cols
            
            # Get the image and label
            img = data[idx]
            label = labels[idx]
            class_name = class_names.get(label, f"Unknown ({label})")
            
            # Process image based on its type and format
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
                
                # Check image dimensions and rearrange if needed
                if img.shape[0] == 3:  # Images likely in CHW format (channels first)
                    img = np.transpose(img, (1, 2, 0))
            
            # If image is a numpy array with shape (channels, height, width)
            elif isinstance(img, np.ndarray) and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            
            # Try different scaling approaches if all images are white
            if np.mean(img) > 0.9:  # If image is too bright
                # Check if values are uint8 but stored as float
                if np.max(img) <= 1.0 and img.dtype in [np.float32, np.float64]:
                    # Images might be normalized to [0,1]
                    img = img
                elif np.max(img) > 1.0 and np.max(img) <= 255:
                    # Images might be in range [0,255]
                    img = img / 255.0
                else:
                    # Try to auto-scale the image to improve visibility
                    # Normalize to [0,1] range
                    img_min = np.min(img)
                    img_max = np.max(img)
                    if img_max > img_min:
                        img = (img - img_min) / (img_max - img_min)
            
            # Make sure values are in valid range for display [0, 1]
            img = np.clip(img, 0, 1)
            
            # Display the image
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"{class_name} (ID: {label})")
            axes[row, col].axis('off')
            
            # Print image statistics for debugging
            print(f"Image {i} - Min: {img.min():.2f}, Max: {img.max():.2f}, Mean: {img.mean():.2f}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Create output directory if it doesn't exist
        os.makedirs("./evaluation/TaskB", exist_ok=True)
        
        # Save the figure
        fig_path = os.path.join("./evaluation/TaskB", f"fixed_original_images_{dataset_names[dataset_idx].replace(' ', '_')}.png")
        plt.savefig(fig_path)
        print(f"Saved figure to {fig_path}")
        plt.close(fig)

if __name__ == "__main__":
    display_original_images_properly()
