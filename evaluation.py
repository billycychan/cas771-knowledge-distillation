import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import traceback
import json
from torchvision import transforms
from tqdm import tqdm

# Import custom modules
try:
    from data_loader_kd import load_all_datasets, custom_collate_fn
    from student_model import StudentModel  # This is the EfficientNetStudent model by default
    from teacher_model_B import set_seed
except ImportError as e:
    print(f"Error importing modules: {e}")
    traceback.print_exc()
    sys.exit(1)

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
        all_images = []
        all_labels = []
        all_predictions = []
        
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
                # Keep track of sample images for display
                if len(all_images) < num_samples_to_display:
                    samples_to_add = min(num_samples_to_display - len(all_images), len(images))
                    all_images.extend(images[:samples_to_add].cpu())
                    all_labels.extend(labels[:samples_to_add].cpu())
                
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                # Keep track of predictions for display samples
                if len(all_predictions) < num_samples_to_display:
                    predictions_to_add = min(num_samples_to_display - len(all_predictions), len(predicted))
                    all_predictions.extend(predicted[:predictions_to_add].cpu())
                
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
        
        print("\nTest Samples with Predictions:")
        print("==============================")
        
        return accuracy, all_images, all_labels, all_predictions, per_class_accuracy
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        return 0, [], [], [], {}

def display_samples(images, labels, predictions, title, class_names=None, output_dir=None):
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
        rows = (num_samples + 4) // 5  # Ceiling division to determine number of rows
        
        fig, axes = plt.subplots(rows, 5, figsize=(15, 3 * rows))
        if rows == 1:
            axes = [axes]  # Make it 2D for consistent indexing
        axes = np.array(axes).flatten()
        
        # Display images with labels
        for i in range(num_samples):
            if i >= len(axes):
                break
                
            img = images[i].permute(1, 2, 0).numpy()
            
            # Denormalize if needed
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            
            # Get class names if available
            true_label = labels[i].item()
            pred_label = predictions[i].item()
            
            true_name = class_names[true_label] if class_names and true_label < len(class_names) else str(true_label)
            pred_name = class_names[pred_label] if class_names and pred_label < len(class_names) else str(pred_label)
            
            # Mark correct/incorrect predictions
            correct = true_label == pred_label
            mark = "✓" if correct else "✗"
            
            axes[i].set_title(f"{true_name} → {pred_name} {mark}", fontsize=10)
            axes[i].axis('off')
            
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.95)
        
        # Save figure
        if output_dir:
            fig_path = os.path.join(output_dir, "sample_predictions.png")
        else:
            fig_path = f"{title.replace(' ', '_')}.png"
            
        print(f"Sample predictions saved to {fig_path}")
        plt.savefig(fig_path)
        
        plt.close(fig)  # Close the figure to avoid display issues
    except Exception as e:
        print(f"Error displaying samples: {e}")
        traceback.print_exc()

def get_class_names(task):
    """
    Get class names for the given task
    """
    if task == 'A':
        return ['pine', 'oak', 'palm', 'willow', 'maple', 'rose', 'tulip', 'daisy', 'iris', 'lily', 'apple', 'orange', 'banana', 'strawberry', 'pear']
    elif task == 'B':
        return ['cat', 'dog', 'horse', 'elephant', 'butterfly', 'chicken', 'sheep', 'spider', 'squirrel', 'cow', 'tiger', 'zebra']
    else:
        return None

def save_results_to_json(results, output_path):
    """
    Save evaluation results to a JSON file
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")
        traceback.print_exc()

def main():
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Evaluate Knowledge Distillation Models')
        parser.add_argument('--task', type=str, default='both', choices=['A', 'B', 'both'], 
                           help='Task to evaluate (A, B, or both)')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
        parser.add_argument('--display_samples', type=int, default=10, 
                           help='Number of sample images to display')
        parser.add_argument('--output_dir', type=str, default='evaluation',
                           help='Directory to save evaluation results')
        args = parser.parse_args()
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        print(f"Using device: {device}")
        
        # Define model paths and classes per task
        model_paths = {
            'A': './best_models/TaskA/student_model_epoch57_acc81.27.pth',
            'B': './best_models/TaskB/student_model_epoch12_acc80.00.pth'
        }
        
        num_classes = {
            'A': 15,  # 15 classes for Task A
            'B': 12   # 12 classes for Task B
        }
        
        # Determine which tasks to evaluate
        tasks_to_evaluate = ['A', 'B'] if args.task == 'both' else [args.task]
        
        # Evaluate each selected task
        for task in tasks_to_evaluate:
            # Create task-specific output directory
            task_output_dir = os.path.join(args.output_dir, f"Task{task}")
            os.makedirs(task_output_dir, exist_ok=True)
            
            print(f"\n{'='*50}")
            print(f"Evaluating Task {task} Model")
            print(f"{'='*50}")
            
            try:
                # Get class names for the task
                class_names = get_class_names(task)
                if class_names:
                    print(f"Test set: {len(class_names) * 100} images")
                    print(f"Classes: {class_names}")
                
                # Load test data
                _, test_loader, num_classes_actual = load_all_datasets(
                    batch_size=args.batch_size, 
                    task=task
                )
                
                # Update num_classes based on actual data if needed
                if num_classes_actual is not None and num_classes_actual > 0:
                    num_classes[task] = num_classes_actual
                
                # Load model
                model = load_model(
                    model_path=model_paths[task], 
                    num_classes=num_classes[task], 
                    device=device
                )
                
                if model is None:
                    print(f"Skipping evaluation for Task {task} due to model loading error.")
                    continue
                
                print(f"Loaded model from {model_paths[task]}")
                print(f"Model's reported accuracy: {81.27 if task == 'A' else 80.0}%")
                
                # Evaluate model
                accuracy, sample_images, sample_labels, sample_predictions, per_class_accuracy = evaluate_model(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                    class_names=class_names,
                    num_samples_to_display=args.display_samples
                )
                
                # Display sample predictions
                if len(sample_images) > 0:
                    print("\nTest Samples with Predictions:")
                    print("==============================")
                    display_samples(
                        images=sample_images[:args.display_samples],
                        labels=sample_labels[:args.display_samples],
                        predictions=sample_predictions[:args.display_samples],
                        title=f"Task {task} Sample Predictions",
                        class_names=class_names,
                        output_dir=task_output_dir
                    )
                else:
                    print("No sample images collected for display")
                
                # Save results to JSON
                results = {
                    "task": task,
                    "model_path": model_paths[task],
                    "overall_accuracy": accuracy,
                    "per_class_accuracy": per_class_accuracy
                }
                
                save_results_to_json(results, os.path.join(task_output_dir, "results.json"))
                
            except Exception as e:
                print(f"Error evaluating Task {task}: {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()