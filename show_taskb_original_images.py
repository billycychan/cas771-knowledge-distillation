import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from data_loader_kd import load_data

def display_original_images():
    """
    Display the original images from the TaskB datasets without any transformations or normalization
    """
    # Load data directly from each validation file
    val_files = [
        './data/TaskB/val_dataB_model_1.pth',  # Mammal dataset
        './data/TaskB/val_dataB_model_2.pth',  # African Animal dataset
        './data/TaskB/val_dataB_model_3.pth',  # Canidae dataset
    ]
    
    # Class mappings based on the user-provided information
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
    
    for dataset_idx, val_file in enumerate(val_files):
        print(f"Loading data from {val_file}")
        data, labels, _ = load_data(val_file, task='B')
        
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
            
            # If image is a torch tensor, convert to numpy
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
            
            # If image is already a numpy array with shape (channels, height, width)
            elif isinstance(img, np.ndarray) and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            
            # Make sure values are in valid range for display [0, 1]
            img = np.clip(img, 0, 1)
            
            # Display the image
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"{class_name} (ID: {label})")
            axes[row, col].axis('off')
            
            # Print image statistics for debugging
            print(f"Image stats - Min: {img.min():.2f}, Max: {img.max():.2f}, Mean: {img.mean():.2f}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Create output directory if it doesn't exist
        os.makedirs("./evaluation/TaskB", exist_ok=True)
        
        # Save the figure
        fig_path = os.path.join("./evaluation/TaskB", f"original_images_{dataset_names[dataset_idx].replace(' ', '_')}.png")
        plt.savefig(fig_path)
        print(f"Saved figure to {fig_path}")
        plt.close(fig)

if __name__ == "__main__":
    display_original_images()
