import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms
import collections

# Import data loader
from data_loader_kd import load_data, load_all_datasets

def get_global_mapping(task='A'):
    """Get the global label mapping used by the data loader"""
    # Define fixed global mappings based on what we observed in previous outputs
    if task == 'A':
        # This is the mapping we saw in the previous runs
        mapping_dict = {0: 0, 47: 1, 51: 2, 52: 3, 53: 4, 54: 5, 56: 6, 57: 7, 59: 8, 62: 9, 70: 10, 82: 11, 83: 12, 92: 13, 96: 14}
        return mapping_dict, "Using hardcoded mapping for TaskA"
    else:
        # For any other task, we'll try to get it from the load_all_datasets function
        # We'll temporarily redirect print output since load_all_datasets prints a lot of info
        import sys
        from io import StringIO
        
        # Redirect stdout
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        try:
            # This will trigger the loading process and print the mapping
            batch_size = 1  # Minimal batch size since we're just getting the mapping
            load_all_datasets(batch_size=batch_size, task=task)
            
            # Extract the global mapping from the output
            output = mystdout.getvalue()
            mapping_lines = [line for line in output.split('\n') if "Global label mapping" in line]
            
            if mapping_lines:
                # Found a mapping line
                mapping_line = mapping_lines[0]
                mapping_str = mapping_line.split('Global label mapping: ')[1].strip()
                
                # Parse the mapping manually instead of using eval
                mapping_dict = {}
                # Remove the curly braces
                mapping_str = mapping_str.strip('{}')
                # Split by commas
                pairs = mapping_str.split(',')
                for pair in pairs:
                    if ':' in pair:
                        key, value = pair.split(':')
                        mapping_dict[int(key.strip())] = int(value.strip())
                
                return mapping_dict, output
            else:
                # No mapping found, return empty dict
                return {}, output
        except Exception as e:
            print(f"Error extracting mapping: {e}")
            return {}, f"Error: {e}"
        finally:
            # Restore stdout
            sys.stdout = old_stdout

def display_raw_data_with_mapping(data_path, task='A', output_dir=None, class_names=None):
    """
    Load and display raw images from a dataset file with correct class name mapping
    """
    print(f"Loading raw data from {data_path}")
    
    # Get the global mapping used by the data loader
    global_mapping, mapping_output = get_global_mapping(task)
    print(f"Global mapping from data_loader_kd: {global_mapping}")
    
    # Invert the mapping to go from original label to global index
    orig_to_global = global_mapping
    
    # Load the raw data
    data, labels, _ = load_data(data_path, task)
    
    print(f"Loaded {len(data)} images with {len(labels)} labels")
    
    # Print mapping between original labels and class names
    print("\nOriginal label to class name mapping:")
    for orig_label in sorted(set(labels)):
        if orig_label in orig_to_global:
            global_idx = orig_to_global[orig_label]
            if class_names and global_idx < len(class_names):
                print(f"  Original label {orig_label} maps to global index {global_idx} ({class_names[global_idx]})")
            else:
                print(f"  Original label {orig_label} maps to global index {global_idx}")
        else:
            print(f"  Warning: Original label {orig_label} not found in global mapping!")
    
    # Count frequency of each mapped label
    mapped_labels = []
    for label in labels:
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label in orig_to_global:
            mapped_labels.append(orig_to_global[label])
    
    label_counts = collections.Counter(mapped_labels)
    print("\nMapped label distribution:")
    for mapped_label, count in sorted(label_counts.items()):
        if class_names and mapped_label < len(class_names):
            print(f"  {mapped_label} ({class_names[mapped_label]}): {count} images")
        else:
            print(f"  {mapped_label}: {count} images")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine how many images to display per class
    max_per_class = 2
    samples_per_class = {}
    
    for i in range(len(data)):
        orig_label = labels[i]
        if isinstance(orig_label, torch.Tensor):
            orig_label = orig_label.item()
        
        if orig_label in orig_to_global:
            mapped_label = orig_to_global[orig_label]
            
            if mapped_label not in samples_per_class:
                samples_per_class[mapped_label] = []
            
            if len(samples_per_class[mapped_label]) < max_per_class:
                samples_per_class[mapped_label].append((i, orig_label, mapped_label))
    
    # Flatten the list of samples
    display_samples = []
    for mapped_label in sorted(samples_per_class.keys()):
        display_samples.extend(samples_per_class[mapped_label])
    
    num_images = len(display_samples)
    
    # Create a grid of subplots
    rows = (num_images + 4) // 5
    fig, axes = plt.subplots(rows, 5, figsize=(15, 3 * rows))
    if rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()
    
    # Display each image with its label
    for i, (img_idx, orig_label, mapped_label) in enumerate(display_samples):
        img = data[img_idx]
        
        # Convert to numpy array if it's a tensor
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
        
        # If image is grayscale (1 channel), convert to RGB
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        
        # Denormalize if needed using ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Display the image
        axes[i].imshow(img)
        
        # Set the title to the label name if provided
        if class_names and mapped_label < len(class_names):
            class_name = class_names[mapped_label]
        else:
            class_name = f"Class {mapped_label}"
        
        # Add a detailed title
        title = f"Orig: {orig_label}\nMapped: {mapped_label} ({class_name})"
        axes[i].set_title(title, fontsize=9)
        
        # Add a text box with class information in the corner of the image
        class_text = f"{class_name}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        axes[i].text(0.05, 0.95, class_text, transform=axes[i].transAxes, fontsize=9,
                   verticalalignment='top', bbox=props)
        
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    # Set the title for the entire figure
    dataset_name = os.path.basename(data_path)
    plt.suptitle(f"Images from {dataset_name} with Label Mapping", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure if an output directory is provided
    if output_dir:
        fig_path = os.path.join(output_dir, f"mapped_data_{os.path.basename(data_path).replace('.pth', '')}.png")
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
        display_raw_data_with_mapping(path, 'A', output_dir, class_names_A)
        print("\n")
