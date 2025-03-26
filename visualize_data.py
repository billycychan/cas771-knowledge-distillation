import os
import torch
import numpy as np
from PIL import Image
from data_loader_kd import load_data

def save_images(data_path, output_dir, model_id):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    data, labels, _ = load_data(data_path, task='B')
    
    # Save each image
    for i in range(min(len(data), 1000)):  # Save first 10 images as sample
        img_data = data[i].transpose(1, 2, 0)  # Convert from CHW to HWC
        # Normalize to 0-255 range
        img_data = ((img_data - img_data.min()) * 255 / (img_data.max() - img_data.min())).astype(np.uint8)
        img = Image.fromarray(img_data)
        img.save(os.path.join(output_dir, f'model{model_id}_img{i}_label{labels[i]}.png'))

def main():
    base_dir = '/home/tmp/workspace/merge_kd/data/TaskB'
    output_dir = '/home/tmp/workspace/merge_kd/data/visualized_images'
    
    # Process each model's data
    for i in range(1, 4):
        data_path = os.path.join(base_dir, f'train_dataB_model_{i}.pth')
        save_images(data_path, output_dir, i)

if __name__ == '__main__':
    main()
