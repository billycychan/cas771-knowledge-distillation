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

def evaluate_model(model, dataloader, device, class_names=None, num_samples_to_display=15):
    """
    Evaluate model accuracy and display sample predictions with detailed metrics
    """
    try:
        model.eval()
        correct = 0
        total = 0
        all_images = []
        all_labels = []
        all_predictions = []
        all_source_ids = []
        
        # For per-class accuracy
        class_correct = {}
        class_total = {}
        if class_names:
            for i, name in enumerate(class_names):
                class_correct[i] = 0
                class_total[i] = 0
        
        # For collecting diverse samples
        samples_per_class = {}
        if class_names:
            for i in range(len(class_names)):
                samples_per_class[i] = []
        
        print("Evaluating: ", end="")
        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                # Store samples for each class (up to 3 per class)
                for i in range(len(labels)):
                    label = labels[i].item()
                    if label in samples_per_class and len(samples_per_class[label]) < 3:
                        # Get source information if available
                        source_id = None
                        
                        # Get original images from our organized samples by class
                        if hasattr(dataloader, 'teacher_model_mapping'):
                            source_id = dataloader.teacher_model_mapping.get(label)
                            
                        samples_per_class[label].append({
                            'image': images[i].cpu(),
                            'label': label,
                            'prediction': predicted[i].item(),
                            'source_id': source_id
                        })
                
                # Update overall accuracy
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
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
        
        # Collect diverse samples from different classes
        for class_idx, samples in samples_per_class.items():
            for sample in samples:
                all_images.append(sample['image'])
                all_labels.append(sample['label'])
                all_predictions.append(sample['prediction'])
                all_source_ids.append(sample.get('source_id'))
        
        print("\nTest Samples with Predictions:")
        print("==============================")
        print(f"Collected {len(all_images)} samples from {len(samples_per_class)} classes")
        
        return accuracy, all_images, all_labels, all_predictions, all_source_ids, per_class_accuracy
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        return 0, [], [], [], {}

def display_samples(images, labels, predictions, title, class_names=None, output_dir=None, random_selection=True, source_ids=None):
    """
    Display sample images with their true and predicted labels
    """
    try:
        if len(images) == 0:
            print("No images to display")
            return
        
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        num_samples = len(images)
        
        # Print debug info about original samples
        print(f"Original data: {num_samples} samples")
        label_distribution = {}
        for i in range(min(10, num_samples)):
            label_val = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
            label_name = class_names[label_val] if class_names and label_val < len(class_names) else str(label_val)
            if label_val not in label_distribution:
                label_distribution[label_val] = 0
            label_distribution[label_val] += 1
            print(f"Sample {i}: Label={label_val} ({label_name})")
        
        print("Label distribution in original samples:")
        for label, count in label_distribution.items():
            label_name = class_names[label] if class_names and label < len(class_names) else str(label)
            print(f"  {label_name}: {count} samples")
        
        # Randomly select samples if requested and if we have more than we need
        if random_selection and num_samples > 10:
            # Use the current time for truly random selection
            np.random.seed(int(time.time()))
            
            # Simple random selection of 10 samples without ensuring class diversity
            selected_indices = np.random.choice(range(num_samples), size=min(10, num_samples), replace=False)
            selected_indices = sorted(selected_indices)  # Sort for consistent display order
            
            print(f"Randomly selected {len(selected_indices)} samples")
            
            # For informational purposes, identify the classes that were selected
            unique_classes = set([labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i] for i in range(len(labels))])
            print(f"Found {len(unique_classes)} unique classes in original samples")
            
            # Get the selected images, labels, predictions and source_ids
            selected_images = [images[i] for i in selected_indices]
            selected_labels = [labels[i] for i in selected_indices]
            selected_predictions = [predictions[i] for i in selected_indices]
            
            # Also get the source_ids if available
            selected_source_ids = None
            if source_ids:
                selected_source_ids = [source_ids[i] for i in selected_indices]
            
            # Update variables for display
            images = selected_images
            labels = selected_labels
            predictions = selected_predictions
            source_ids = selected_source_ids
            num_samples = len(images)
            
            # Print selected classes for verification
            print("\nSelected samples:")
            for i, idx in enumerate(selected_indices):
                label_val = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i] 
                label_name = class_names[label_val] if class_names and label_val < len(class_names) else str(label_val)
                print(f"  {i}: Index {idx} - Class {label_val} ({label_name})")
                
            # Print selected classes for verification with detailed information
            selected_classes = [labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i] for i in range(len(labels))]
            class_names_list = [class_names[cls] if cls < len(class_names) else f"Unknown({cls})" for cls in selected_classes]
            print(f"Selected samples from classes: {class_names_list}")
            
            # Create a confusion matrix for the selected samples
            correct = 0
            incorrect = 0
            confusion = {}
            for i in range(len(labels)):
                true_label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
                pred_label = predictions[i].item() if isinstance(predictions[i], torch.Tensor) else predictions[i]
                
                true_name = class_names[true_label] if true_label < len(class_names) else f"Unknown({true_label})"
                pred_name = class_names[pred_label] if pred_label < len(class_names) else f"Unknown({pred_label})"
                
                key = f"{true_name} -> {pred_name}"
                if key not in confusion:
                    confusion[key] = 0
                confusion[key] += 1
                
                if true_label == pred_label:
                    correct += 1
                else:
                    incorrect += 1
            
            print(f"\nSelected samples accuracy: {correct}/{len(labels)} ({100*correct/len(labels):.1f}%)")
            print("Confusion distribution:")
            for pair, count in confusion.items():
                print(f"  {pair}: {count}")
        
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
                
            # Get normalized image data and properly denormalize it back to original values
            img = images[i].permute(1, 2, 0).cpu().numpy()
            
            # The images were normalized with ImageNet mean and std during data loading
            # We need to denormalize to see the original colors
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
            # Debug print image stats to check for anomalies
            if i < 3:  # Just print a few to avoid console spam
                print(f"Image {i} - Min: {img.min():.2f}, Max: {img.max():.2f}, Mean: {img.mean():.2f}")
            
            axes[i].imshow(img)
            
            # Get class names if available
            true_label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
            pred_label = predictions[i].item() if isinstance(predictions[i], torch.Tensor) else predictions[i]
            
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
            if source_ids and i < len(source_ids) and source_ids[i] is not None:
                source_info = f" (Teacher {source_ids[i]+1})"
                
            # Create a title with clear emphasis on the true label
            title_text = f"True: {true_name}{source_info}\nPred: {pred_name} {mark}"
            
            # Add custom title with color for match/mismatch
            axes[i].set_title(title_text, fontsize=9, pad=5, color=color)
            
            # Add a colored border around the image to indicate correct/incorrect prediction
            if not correct:
                # Add a red border for incorrect predictions
                axes[i].spines['bottom'].set_color('red')
                axes[i].spines['top'].set_color('red') 
                axes[i].spines['right'].set_color('red')
                axes[i].spines['left'].set_color('red')
                axes[i].spines['bottom'].set_linewidth(2)
                axes[i].spines['top'].set_linewidth(2) 
                axes[i].spines['right'].set_linewidth(2)
                axes[i].spines['left'].set_linewidth(2)
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

def get_class_names_taska():
    """
    Get class names for TaskA based on the correct mapping from the original datasets
    
    The global mapping in data_loader_kd.py is:
    {0: 0, 47: 1, 51: 2, 52: 3, 53: 4, 54: 5, 56: 6, 57: 7, 59: 8, 62: 9, 70: 10, 82: 11, 83: 12, 92: 13, 96: 14}
    
    The original dataset labels are:
    - Model 1 (Trees): maple_tree (47), oak_tree (52), palm_tree (56), pine_tree (59), willow (96)
    - Model 2 (Flowers): orchid (54), poppy (62), rose (70), sunflower (82), tulip (92)
    - Model 3 (Fruits and Vegetables): apple (0), mushroom (51), orange (53), pear (57), sweet_pepper (83)
    """
    return [
        "apple",         # 0  - From model3 (index 0)
        "maple_tree",    # 1  - From model1 (index 47)
        "mushroom",      # 2  - From model3 (index 51)
        "oak_tree",      # 3  - From model1 (index 52)
        "orange",        # 4  - From model3 (index 53)
        "orchid",        # 5  - From model2 (index 54)
        "palm_tree",     # 6  - From model1 (index 56)
        "pear",          # 7  - From model3 (index 57)
        "pine_tree",     # 8  - From model1 (index 59)
        "poppy",         # 9  - From model2 (index 62)
        "rose",          # 10 - From model2 (index 70)
        "sunflower",     # 11 - From model2 (index 82)
        "sweet_pepper",  # 12 - From model3 (index 83)
        "tulip",         # 13 - From model2 (index 92)
        "willow"         # 14 - From model1 (index 96)
    ]

def load_custom_test_data(batch_size=16):
    """
    Load test data for TaskA using the existing data loader function
    Also load the raw images directly from the original datasource for display
    """
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from data_loader_kd import load_all_datasets
    
    # First, load test data using the existing function which handles the data properly
    print("Loading test data using the data_loader_kd module...")
    _, test_loader, num_classes = load_all_datasets(batch_size=batch_size, task='A')
    
    # Store the original data source samples separately by original class
    # This will help us retrieve the correct original image based on class
    original_samples_by_class = {}
    
    # The mapping from original indices to our global indices
    # This is the key to correctly pairing original images with our processed datasets
    original_to_global_mapping = {
        # Model 1 (Trees)
        47: 1,  # maple_tree
        52: 3,  # oak_tree
        56: 6,  # palm_tree
        59: 8,  # pine_tree
        96: 14, # willow
        
        # Model 2 (Flowers)
        54: 5,  # orchid
        62: 9,  # poppy
        70: 10, # rose
        82: 11, # sunflower
        92: 13, # tulip
        
        # Model 3 (Fruits and Vegetables)
        0: 0,   # apple
        51: 2,  # mushroom
        53: 4,  # orange
        57: 7,  # pear
        83: 12  # sweet_pepper
    }
    
    # Map from global index to teacher model
    teacher_model_mapping = {
        # Model 1 (Trees)
        1: 0,  # maple_tree - Teacher 1
        3: 0,  # oak_tree - Teacher 1
        6: 0,  # palm_tree - Teacher 1
        8: 0,  # pine_tree - Teacher 1
        14: 0, # willow - Teacher 1
        
        # Model 2 (Flowers)
        5: 1,  # orchid - Teacher 2
        9: 1,  # poppy - Teacher 2
        10: 1, # rose - Teacher 2
        11: 1, # sunflower - Teacher 2
        13: 1, # tulip - Teacher 2
        
        # Model 3 (Fruits and Vegetables)
        0: 2,  # apple - Teacher 3
        2: 2,  # mushroom - Teacher 3
        4: 2,  # orange - Teacher 3
        7: 2,  # pear - Teacher 3
        12: 2  # sweet_pepper - Teacher 3
    }
    
    # Define paths to the original data files
    data_paths = [
        'data/TaskA/Model1_trees_superclass/model1_test_supercls.pth',
        'data/TaskA/Model2_flowers_superclass/model2_test_supercls.pth',
        'data/TaskA/Model3_fruit+veg_superclass/model3_test_supercls.pth'
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
            class_name = get_class_names_taska()[class_idx]
            count = len(original_samples_by_class[class_idx])
            print(f"  Class {class_idx} ({class_name}): {count} images")
            
        # Store the organized samples in the test_loader
        test_loader.original_samples_by_class = original_samples_by_class
        test_loader.has_original_images = True
        test_loader.teacher_model_mapping = teacher_model_mapping
    else:
        print("Warning: Could not load complete set of original images for display")
        print(f"Found images for {len(original_samples_by_class)} out of {num_classes} classes")
        test_loader.original_samples_by_class = original_samples_by_class
        test_loader.has_original_images = bool(original_samples_by_class)
        test_loader.teacher_model_mapping = teacher_model_mapping
    
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
        model_path = './best_models/TaskA/student_model_epoch57_acc81.27.pth'
        
        # Create output directory
        output_dir = './evaluation/TaskA'
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data with smaller batch size
        test_loader, num_classes = load_custom_test_data(batch_size=16)
        
        # Get class names
        class_names = get_class_names_taska()
        
        # Load model
        model = load_model(model_path, num_classes, device)
        if model is None:
            print("Failed to load model. Exiting.")
            return
        
        # Evaluate model
        accuracy, sample_images, sample_labels, sample_predictions, sample_source_ids, per_class_accuracy = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=class_names,
            num_samples_to_display=10
        )
        
        # Display sample predictions with random selection
        if len(sample_images) > 0:
            display_samples(
                images=sample_images,
                labels=sample_labels,
                predictions=sample_predictions,
                title="TaskA Sample Predictions",
                class_names=class_names,
                output_dir=output_dir,
                random_selection=True,
                source_ids=sample_source_ids
            )
        
        # Save results to JSON
        results = {
            "task": "A",
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
