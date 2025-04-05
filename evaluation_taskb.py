import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import traceback
import json
import time
from torchvision import transforms
from tqdm import tqdm

# Import custom modules
from data_loader_kd import load_data, CAS771Dataset, custom_collate_fn
from student_model import StudentModel  # This is the EfficientNetStudent model by default
from torch.utils.data import DataLoader

def load_model(model_path, num_classes, device):
    """
    Load a trained student model from checkpoint
    """
    try:
        model = StudentModel(num_classes=num_classes, pretrained=False)
        
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Error: Model checkpoint not found at {model_path}")
            return None
            
        model.to(device)
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def evaluate_model(model, dataloader, device, class_names=None, num_samples_to_display=10):
    """
    Evaluate model accuracy and display sample predictions with detailed metrics
    """
    try:
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        # For per-class accuracy
        class_correct = {}
        class_total = {}
        if class_names:
            for i, name in enumerate(class_names):
                class_correct[i] = 0
                class_total[i] = 0
        
        print("Evaluating: ", end="")
        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                # Update overall accuracy
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions and labels for analysis
                all_predictions.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
                # Update per-class accuracy
                if class_names:
                    for i in range(len(labels)):
                        label = labels[i].item()
                        if label in class_correct:
                            class_total[label] += 1
                            if predicted[i].item() == label:
                                class_correct[label] += 1
        
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"\nOverall Accuracy: {accuracy:.2f}%")
        
        # Calculate per-class accuracy
        per_class_accuracy = {}
        if class_names:
            print("\nPer-class Accuracy:")
            for i, name in enumerate(class_names):
                if i in class_total and class_total[i] > 0:
                    acc = 100 * class_correct[i] / class_total[i]
                    per_class_accuracy[name] = acc
                    print(f"  {name}: {acc:.2f}%")
                else:
                    per_class_accuracy[name] = 0
                    print(f"  {name}: No samples")
        
        print("\nTest Samples with Predictions:")
        print("==============================")
        
        return accuracy, all_labels, all_predictions, per_class_accuracy
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        return 0, [], [], {}

def display_samples(dataloader, model, device, title, class_names=None, output_dir=None, random_selection=True, num_samples=10):
    """
    Display sample images with their true and predicted labels
    """
    try:
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get samples directly from the dataloader
        model.eval()
        
        # We'll collect samples by class to ensure diversity
        samples_by_class = {}
        all_classes = set()
        
        # First, collect samples from each class
        print("Collecting samples from test loader...")
        with torch.no_grad():
            for batch_idx, (batch_images, batch_labels, _) in enumerate(dataloader):
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                
                # Get predictions
                outputs = model(batch_images)
                _, batch_predictions = torch.max(outputs, 1)
                
                # Process each image in the batch
                for i in range(len(batch_labels)):
                    label = batch_labels[i].item()
                    all_classes.add(label)
                    
                    if label not in samples_by_class:
                        samples_by_class[label] = []
                    
                    # Only collect a few samples per class to avoid bias
                    if len(samples_by_class[label]) < 5:  # Limit to 5 samples per class
                        # Get the index of this sample in the dataset
                        sample_idx = batch_idx * dataloader.batch_size + i
                        
                        # Store both the processed image (for model) and the original image (for display)
                        original_img = None
                        source_id = None
                        
                        # Get original images from our organized samples by class
                        if hasattr(dataloader, 'original_samples_by_class') and hasattr(dataloader, 'has_original_images') and dataloader.has_original_images:
                            # For each class, we have a list of original samples
                            if label in dataloader.original_samples_by_class and dataloader.original_samples_by_class[label]:
                                # Choose a random sample for this class
                                import random
                                original_sample = random.choice(dataloader.original_samples_by_class[label])
                                
                                # Extract the original image and source information
                                original_img = original_sample['image']
                                source_id = original_sample['source_id']
                                orig_label = original_sample['original_label']
                                
                                # Print debug info for the first few samples
                                if len(samples_by_class[label]) < 2:
                                    class_name = get_class_names_taskb()[label] if label < len(get_class_names_taskb()) else str(label)
                                    print(f"Retrieved original image for class {label} ({class_name}) from teacher {source_id+1}, original label {orig_label}")
                            
                        samples_by_class[label].append({
                            'image': batch_images[i].cpu(),
                            'original_image': original_img,
                            'label': label,
                            'prediction': batch_predictions[i].item(),
                            'dataset_index': sample_idx,
                            'source_id': source_id
                        })
                
                # If we have collected samples from all classes, we can stop
                if len(all_classes) == len(class_names) and all(len(samples) >= 2 for samples in samples_by_class.values()):
                    break
        
        # Now select a diverse set of samples for display
        selected_samples = []
        
        # First, try to include one sample from each class, but limit to 10 total
        available_classes = sorted(samples_by_class.keys())
        
        # If we have more than 10 classes, randomly select 10
        import random
        if len(available_classes) > 10:
            random.shuffle(available_classes)
            available_classes = available_classes[:10]
        
        # Add one sample from each selected class
        for label in available_classes:
            if samples_by_class[label]:
                # Randomly select one sample from this class
                sample_index = random.randrange(len(samples_by_class[label]))
                sample = samples_by_class[label][sample_index]
                selected_samples.append(sample)
                
                # Remove this sample to avoid duplicates (using index to avoid tensor comparison)
                samples_by_class[label].pop(sample_index)
                
                # Stop if we've reached 10 samples
                if len(selected_samples) >= 10:
                    break
        
        # If we need more samples to reach exactly 10, add more randomly
        if len(selected_samples) < 10:
            all_remaining_samples = []
            for samples in samples_by_class.values():
                all_remaining_samples.extend(samples)
                
            # Shuffle remaining samples
            random.shuffle(all_remaining_samples)
            
            # Add more samples until we reach exactly 10 or run out
            while len(selected_samples) < 10 and all_remaining_samples:
                selected_samples.append(all_remaining_samples.pop())
        
        # Ensure we have exactly 10 samples (or fewer if not enough data)
        selected_samples = selected_samples[:10]
        
        # Extract the data from the selected samples
        images = [sample['image'] for sample in selected_samples]
        original_images = [sample['original_image'] for sample in selected_samples]
        source_ids = [sample['source_id'] for sample in selected_samples]
        labels = [sample['label'] for sample in selected_samples]
        predictions = [sample['prediction'] for sample in selected_samples]
        dataset_indices = [sample['dataset_index'] for sample in selected_samples]
        
        num_samples = len(images)
        print(f"Collected {num_samples} samples for display")
        
        if num_samples == 0:
            print("No images to display")
            return
        
        # Print debug info about collected samples
        print(f"Original data: {num_samples} samples")
        label_distribution = {}
        for i in range(min(10, num_samples)):
            label_val = labels[i]
            label_name = class_names[label_val] if class_names and label_val < len(class_names) else str(label_val)
            if label_val not in label_distribution:
                label_distribution[label_val] = 0
            label_distribution[label_val] += 1
            # Handle potential None values in source_ids
            source = ""
            if i < len(source_ids) and source_ids[i] is not None:
                source = f"(from teacher model {source_ids[i]+1})"
            print(f"Sample {i}: Label={label_val} ({label_name}) {source}")
        
        print("Label distribution in collected samples:")
        for label, count in label_distribution.items():
            label_name = class_names[label] if class_names and label < len(class_names) else str(label)
            print(f"  {label_name}: {count} samples")
        
        # Determine grid layout
        rows = (num_samples + 4) // 5  # Ceiling division to determine number of rows
        
        # Create figure with more space for titles
        fig, axes = plt.subplots(rows, 5, figsize=(15, 4 * rows))
        if rows == 1:
            axes = [axes]  # Make it 2D for consistent indexing
        axes = np.array(axes).flatten()
        
        # Display images with labels
        for i in range(num_samples):
            if i >= len(axes):
                break
                
            # Use original image if available, otherwise use the normalized image
            if i < len(original_images) and original_images[i] is not None:
                # The original images are in [H, W, C] format with values 0-255
                # But may be stored as a tensor, so convert to numpy first
                if isinstance(original_images[i], torch.Tensor):
                    # Convert tensor to numpy, handling different tensor formats
                    if original_images[i].ndim == 4 and original_images[i].shape[0] == 1:
                        # Handle batch dimension if present
                        img = original_images[i][0].cpu().numpy()
                    elif original_images[i].ndim == 3 and original_images[i].shape[0] == 3:
                        # Handle [C, H, W] format
                        img = original_images[i].permute(1, 2, 0).cpu().numpy()
                    else:
                        # Assume [H, W, C] format
                        img = original_images[i].cpu().numpy()
                else:
                    # Already a numpy array
                    img = original_images[i]
                
                # Ensure the image is properly formatted for display
                if img.dtype != np.uint8:
                    # Convert from float to uint8 if needed
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                
                # Clip values to valid range and ensure [H, W, C] format
                if img.shape[-1] != 3 and img.shape[0] == 3:
                    # If in [C, H, W] format, convert to [H, W, C]
                    img = np.transpose(img, (1, 2, 0))
                
                img = np.clip(img, 0, 255).astype(np.uint8)
            else:
                # Get image data and convert to numpy for display
                img = images[i].permute(1, 2, 0).cpu().numpy()
                
                # These images are normalized with the dataset mean and std
                # Use the dataset stats from the data loader to denormalize
                mean = np.array([0.5118, 0.5361, 0.6039])
                std = np.array([0.2410, 0.2320, 0.2406])
                
                # Denormalize the image to get back to original pixel values
                img = img * std + mean
                
                # Convert to uint8 for proper display
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            
            # Print image stats for debugging
            if i < 3:
                print(f"Image {i} - Min: {img.min():.2f}, Max: {img.max():.2f}, Mean: {img.mean():.2f}, Shape: {img.shape}")
            
            # Display the image with consistent settings for all images
            axes[i].imshow(img, interpolation='nearest')
            
            # Remove axis for cleaner display
            axes[i].axis('off')
            
            # Get class names if available
            true_label = labels[i]
            pred_label = predictions[i]
            
            true_name = class_names[true_label] if class_names and true_label < len(class_names) else str(true_label)
            pred_name = class_names[pred_label] if class_names and pred_label < len(class_names) else str(pred_label)
            
            # Mark correct/incorrect predictions
            correct = true_label == pred_label
            mark = "✓" if correct else "✗"
            
            # Create a more distinct title with clear separation between true and predicted labels
            # Add color coding: green for correct, red for incorrect
            color = 'green' if correct else 'red'
            
            # Get source info if available (which teacher model this came from)
            source_info = ""
            if i < len(source_ids) and source_ids[i] is not None:
                source_info = f" (Teacher {source_ids[i]+1})"
                
            # Create a title with clear emphasis on the true label
            title_text = f"True: {true_name}{source_info}\nPred: {pred_name} {mark}"
            
            # Add custom title with color for match/mismatch
            axes[i].set_title(title_text, fontsize=9, pad=5, color=color)
            
            # Add a colored border around the image to indicate correct/incorrect prediction
            if not correct:
                # Add a red border for incorrect predictions
                for spine in axes[i].spines.values():
                    spine.set_color('red')
                    spine.set_linewidth(2)
            
            axes[i].axis('off')
            
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
            
        # Adjust layout to prevent title overlap
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at top for suptitle
        plt.suptitle(title, fontsize=16, y=0.98)  # Position title higher
        
        # Save figure
        if output_dir:
            fig_path = os.path.join(output_dir, "sample_predictions.png")
        else:
            fig_path = f"{title.replace(' ', '_')}.png"
            
        print(f"Sample predictions saved to {fig_path}")
        plt.savefig(fig_path, bbox_inches='tight')  # Use tight bbox to avoid cutting off content
        
        plt.close(fig)  # Close the figure to avoid display issues
    except Exception as e:
        print(f"Error displaying samples: {e}")
        traceback.print_exc()


def get_class_names_taskb():
    """
    Get class names for TaskB based on the correct mapping from the original datasets
    
    The global mapping for TaskB is:
    {24: 0, 34: 1, 80: 2, 124: 3, 125: 4, 130: 5, 135: 6, 137: 7, 159: 8, 173: 9, 201: 10, 202: 11}
    
    These are CIFAR-100 classes.
    """
    return [
        "African_elephant",    # 0 - Original index 24
        "hyena",              # 1 - Original index 34
        "zebra",              # 2 - Original index 80
        "collie",             # 3 - Original index 124
        "golden_retriever",   # 4 - Original index 125
        "boxer",              # 5 - Original index 130
        "patas",              # 6 - Original index 135
        "baboon",             # 7 - Original index 137
        "Arctic_fox",         # 8 - Original index 159
        "Chihuahua",          # 9 - Original index 173
        "lynx",               # 10 - Original index 201
        "African_hunting_dog" # 11 - Original index 202
    ]

def load_custom_test_data(batch_size=16):
    """
    Load test data for TaskB using the existing data loader function
    Also load the raw images directly from the original datasource for display
    """
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from data_loader_kd import load_data, CAS771Dataset, custom_collate_fn, load_all_datasets
    
    # First, load test data using the existing function which handles the data properly
    print("Loading test data using the data_loader_kd module...")
    _, test_loader, num_classes = load_all_datasets(batch_size=batch_size, task='B')
    
    # Store the original data source samples separately by original class
    # This will help us retrieve the correct original image based on class
    original_samples_by_class = {}
    
    # The mapping from CIFAR-100 original indices to our global indices
    # This is the key to correctly pairing original images with our processed datasets
    original_to_global_mapping = {24: 0, 34: 1, 80: 2, 124: 3, 125: 4, 130: 5, 
                                135: 6, 137: 7, 159: 8, 173: 9, 201: 10, 202: 11}
    
    # Define paths to the original data files
    data_paths = [
        './data/TaskB/val_dataB_model_1.pth',
        './data/TaskB/val_dataB_model_2.pth',
        './data/TaskB/val_dataB_model_3.pth'
    ]
    
    print("Loading original image data from source files...")
    total_samples = 0
    
    for source_id, path in enumerate(data_paths):
        try:
            # Load the raw data file
            raw_data = torch.load(path)
            data = raw_data['data']  # Original format: [B, H, W, C] with values 0-255
            labels = raw_data['labels']
            
            # Process each image and organize by global class label
            for i in range(len(data)):
                orig_label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
                
                # Map to global label if it's one of our classes
                if orig_label in original_to_global_mapping:
                    global_label = original_to_global_mapping[orig_label]
                    
                    # Initialize the class entry if needed
                    if global_label not in original_samples_by_class:
                        original_samples_by_class[global_label] = []
                    
                    # Store the original image, original label, and source
                    original_samples_by_class[global_label].append({
                        'image': data[i],
                        'original_label': orig_label,
                        'global_label': global_label,
                        'source_id': source_id,
                        'source_path': path
                    })
                    total_samples += 1
            
            print(f"Loaded and processed {len(data)} samples from {path}")
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    
    # Check if we have collected images for all classes
    if original_samples_by_class and len(original_samples_by_class) == num_classes:
        print(f"Successfully loaded {total_samples} original images from source files")
        print(f"Images per class:")
        for class_idx in sorted(original_samples_by_class.keys()):
            class_name = get_class_names_taskb()[class_idx]
            count = len(original_samples_by_class[class_idx])
            print(f"  Class {class_idx} ({class_name}): {count} images")
            
        # Store the organized samples in the test_loader
        test_loader.original_samples_by_class = original_samples_by_class
        test_loader.has_original_images = True
    else:
        print("Warning: Could not load complete set of original images for display")
        print(f"Found images for {len(original_samples_by_class)} out of {num_classes} classes")
        test_loader.original_samples_by_class = original_samples_by_class
        test_loader.has_original_images = bool(original_samples_by_class)
    
    print(f"Loaded test data with {num_classes} classes")
    return test_loader, num_classes

def main():
    try:
        # Set device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        print(f"Using device: {device}")
        
        # Define paths
        model_path = './best_models/TaskB/student_model_epoch12_acc80.00.pth'
        
        # Create output directory
        output_dir = './evaluation/TaskB'
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data with smaller batch size
        test_loader, num_classes = load_custom_test_data(batch_size=16)
        
        # Get class names for TaskB
        class_names = get_class_names_taskb()
        
        # Load model
        model = load_model(model_path, num_classes, device)
        if model is None:
            print("Failed to load model. Exiting.")
            return
        
        # Evaluate model
        accuracy, all_labels, all_predictions, per_class_accuracy = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=class_names,
            num_samples_to_display=10
        )
        
        # Display sample predictions with random selection
        display_samples(
            dataloader=test_loader,
            model=model,
            device=device,
            title="TaskB Sample Predictions",
            class_names=class_names,
            output_dir=output_dir,
            random_selection=True,
            num_samples=10  # Limit to exactly 10 predictions
        )
        
        # Save results to JSON
        results = {
            "task": "B",  # Correctly identify this as Task B
            "model_path": model_path,
            "overall_accuracy": accuracy,
            "per_class_accuracy": per_class_accuracy
        }
        
        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {os.path.join(output_dir, 'results.json')}")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
