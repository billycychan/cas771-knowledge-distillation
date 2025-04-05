import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms

# Import data loader
from data_loader_kd import load_data

def display_raw_data(data_path, task='A', output_dir=None, class_names=None):
    """
    Load and display raw images from a dataset file to verify label correctness
    """
    print(f"Loading raw data from {data_path}")
    
    # Load the raw data
    data, labels, mapping = load_data(data_path, task)
    
    print(f"Loaded {len(data)} images with {len(labels)} labels")
    
    # Print detailed label information
    print(f"Label mapping from dataset: {mapping}")
    
    # Count frequency of each label
    label_counts = {}
    for label in labels:
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    
    print("Label distribution:")
    for label, count in sorted(label_counts.items()):
        label_name = class_names[label] if class_names and label < len(class_names) else f"Unknown({label})"
        print(f"  {label} ({label_name}): {count} images")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine how many images to display
    num_images = min(25, len(data))
    
    # Create a grid of subplots
    rows = (num_images + 4) // 5
    fig, axes = plt.subplots(rows, 5, figsize=(15, 3 * rows))
    if rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()
    
    # Display each image with its label
    for i in range(num_images):
        img = data[i]
        label = labels[i]
        
        # Convert to numpy array if it's a tensor
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        
        # If image is grayscale (1 channel), convert to RGB
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        
        # Transpose if needed
        if img.shape[0] == 3 and len(img.shape) == 3:  # If channels first (C, H, W)
            img = np.transpose(img, (1, 2, 0))
        
        # Normalize to [0,1] if needed
        if img.max() > 1.0:
            img = img / 255.0
        
        # Display the image
        axes[i].imshow(img)
        
        # Get the original label value from the dataset
        if isinstance(label, torch.Tensor):
            label = label.item()
            
        # Set the title to the label name if provided
        if class_names and label < len(class_names):
            label_name = class_names[label]
        else:
            label_name = f"Class {label}"
        
        # Add more detailed information to the title
        axes[i].set_title(f"Label {label} ({label_name})\nIdx: {i}", fontsize=10)
        
        # Add a text box with class information in the corner of the image
        class_text = f"{label_name}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        axes[i].text(0.05, 0.95, class_text, transform=axes[i].transAxes, fontsize=9,
                   verticalalignment='top', bbox=props)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    # Set the title for the entire figure
    plt.suptitle(f"Raw Images from {os.path.basename(data_path)}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure if an output directory is provided
    if output_dir:
        fig_path = os.path.join(output_dir, f"raw_data_{os.path.basename(data_path).replace('.pth', '')}.png")
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Saved figure to {fig_path}")
    
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Define class names for TaskA
    class_names_A = ['pine', 'oak', 'palm', 'willow', 'maple', 'rose', 'tulip', 'daisy', 'iris', 'lily', 
                   'apple', 'orange', 'banana', 'strawberry', 'pear']
    
    # Output directory
    output_dir = "./data_inspection"
    
    # Print global label mapping for reference
    print("Global class mapping for TaskA:")
    for idx, name in enumerate(class_names_A):
        print(f"  {idx}: {name}")
    print("\n")
    
    # Inspect data for all three model files in TaskA
    data_paths = [
        './data/TaskA/Model1_trees_superclass/model1_test_supercls.pth',
        './data/TaskA/Model2_flowers_superclass/model2_test_supercls.pth',
        './data/TaskA/Model3_fruit+veg_superclass/model3_test_supercls.pth'
    ]
    
    for path in data_paths:
        print(f"\n{'='*50}")
        print(f"Inspecting {os.path.basename(path)}")
        print(f"{'='*50}")
        display_raw_data(path, 'A', output_dir, class_names_A)
        print("\n")
