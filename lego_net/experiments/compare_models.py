#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compare the three different models:
1. Original LegoTransformer
2. Random Walk-specific LegoTransformer
3. Step Predictor Transformer
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Update imports to use full paths from root package
from lego_net.model.transformer import LegoTransformer
from lego_net.model.step_predictor import StepPredictorTransformer
from lego_net.data.data_generator import generate_brick_sequence, visualize_sequence
from lego_net.utils.training import generate_sequence as generate_sequence_original

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory
output_dir = './output/model_comparison'
os.makedirs(output_dir, exist_ok=True)

# Load the three models
model_paths = {
    'original': './checkpoints/best_model.pt',
    'random_walk': './checkpoints/random_walk_model/best_model.pt',
    'step_predictor': './checkpoints/step_predictor_model/best_model.pt'
}

# Function to load a model
def load_model(model_path, model_class):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Determine if this is a state_dict or full checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Create model with appropriate architecture
        if model_class == LegoTransformer:
            if 'random_walk' in model_path:
                model = model_class(d_model=64, nhead=8, num_layers=3, dim_feedforward=128)
            else:
                model = model_class()
        else:
            model = model_class(d_model=64, nhead=8, num_layers=3, dim_feedforward=128)
            
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

# Load models
models = {}
try:
    models['original'] = load_model(model_paths['original'], LegoTransformer)
    models['random_walk'] = load_model(model_paths['random_walk'], LegoTransformer)
    models['step_predictor'] = load_model(model_paths['step_predictor'], StepPredictorTransformer)
    
    # Check which models were loaded successfully
    available_models = [name for name, model in models.items() if model is not None]
    print(f"Successfully loaded models: {', '.join(available_models)}")
    
except Exception as e:
    print(f"Error loading models: {e}")
    available_models = []

if not available_models:
    print("No models could be loaded. Please train the models first.")
    exit(1)

# Function to evaluate models on test data
def evaluate_models(pattern, num_test_cases=20):
    print(f"\nEvaluating models on {pattern} pattern:")
    
    # Create test data
    sequences = [generate_brick_sequence(pattern, length=10) for _ in range(num_test_cases)]
    test_batch = torch.stack(sequences, dim=0).to(device)
    
    # Get input and target
    inputs = test_batch[:, :-1, :]
    targets = test_batch[:, 1:, :]
    
    # Evaluate each available model
    results = {}
    
    for name in available_models:
        model = models[name]
        
        with torch.no_grad():
            if name == 'step_predictor':
                # Step predictor model returns two outputs
                _, pred_coords = model(inputs)
                mse = torch.mean((pred_coords - inputs[:, :, :]) ** 2).item()
            else:
                # Original and random_walk models
                outputs = model(inputs)
                mse = torch.mean((outputs - targets) ** 2).item()
                
        results[name] = mse
        print(f"  {name} model MSE: {mse:.6f}")
    
    return results

# Function to generate and visualize sequences
def generate_and_visualize(pattern):
    print(f"\nGenerating and comparing {pattern} sequences:")
    
    # Generate a starting sequence
    seed_seq = generate_brick_sequence(pattern, length=5)
    start_brick = seed_seq.unsqueeze(0).to(device)  # [1, 5, 3]
    
    # Generate sequences with each model
    sequences = {}
    
    for name in available_models:
        model = models[name]
        
        with torch.no_grad():
            if name == 'step_predictor':
                # Step predictor model has its own generation method
                generated = model.generate_sequence(start_brick, 15)
            else:
                # Original and random_walk models
                # Use only the first 2 bricks as seed to make it more challenging
                generated = generate_sequence_original(model, start_brick[:, :2, :], 18, device)
            
            sequences[name] = generated.squeeze(0).cpu().numpy()
    
    # Plot sequences
    num_models = len(available_models)
    fig = plt.figure(figsize=(15, 5 * ((num_models + 1) // 2)))
    
    # Ground truth
    ground_truth = seed_seq.numpy()
    ax = fig.add_subplot(num_models+1, 2, 1, projection='3d')
    ax.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
               c=range(len(ground_truth)), cmap='viridis', s=100)
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], 'k--', alpha=0.4)
    for i, (x, y, z) in enumerate(ground_truth):
        ax.text(x, y, z, f" {i}", fontsize=10, color='red')
    ax.set_title(f'Ground Truth Starting Sequence ({pattern})')
    
    # Plot each model's generation
    for i, name in enumerate(available_models):
        ax = fig.add_subplot(num_models+1, 2, i+2, projection='3d')
        seq = sequences[name]
        
        ax.scatter(seq[:, 0], seq[:, 1], seq[:, 2],
                 c=range(len(seq)), cmap='viridis', s=100)
        ax.plot(seq[:, 0], seq[:, 1], seq[:, 2], 'k--', alpha=0.4)
        
        # Show only a few indices to avoid cluttering
        for j in range(0, len(seq), 3):
            x, y, z = seq[j]
            ax.text(x, y, z, f" {j}", fontsize=10, color='red')
            
        ax.set_title(f'{name} Model Generation')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'comparison_{pattern}.png')
    plt.savefig(save_path)
    plt.close(fig)
    
    print(f"  Visualizations saved to {save_path}")
    
    return sequences

# Evaluate on all pattern types
patterns = ['random_walk', 'stack', 'row', 'stair']
all_results = {}

for pattern in patterns:
    all_results[pattern] = evaluate_models(pattern)
    generate_and_visualize(pattern)

# Plot comparison of all models on all patterns
patterns_display = {'random_walk': 'Random Walk', 'stack': 'Stack', 'row': 'Row', 'stair': 'Stair'}
model_colors = {'original': 'blue', 'random_walk': 'green', 'step_predictor': 'red'}

plt.figure(figsize=(12, 6))
bar_width = 0.25
index = np.arange(len(patterns))

for i, model_name in enumerate(available_models):
    mse_values = [all_results[pattern][model_name] for pattern in patterns]
    plt.bar(index + i*bar_width, mse_values, bar_width, 
            label=f'{model_name} Model', color=model_colors.get(model_name, 'gray'))

plt.xlabel('Pattern Type')
plt.ylabel('Mean Squared Error')
plt.title('Model Performance Comparison')
plt.xticks(index + bar_width, [patterns_display[p] for p in patterns])
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'overall_comparison.png'))
plt.close()

print("\nComparison complete. Results saved to", output_dir) 