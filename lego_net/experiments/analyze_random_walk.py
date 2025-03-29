#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to analyze random walk patterns and their prediction difficulty
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Update imports to use full paths from root package
from lego_net.data.data_generator import generate_brick_sequence, visualize_sequence
from lego_net.model.transformer import LegoTransformer
from lego_net.utils.training import generate_sequence

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory
output_dir = './output/random_walk_analysis'
os.makedirs(output_dir, exist_ok=True)

# 1. Generate a set of random walks and analyze their properties
print("Analyzing random walk patterns...")
num_samples = 10
seq_length = 20

# Keep track of statistics
step_sizes = []
z_increments = []
directions = []

# Generate and analyze multiple random walks
for i in range(num_samples):
    # Generate a sequence
    seq = generate_brick_sequence('random_walk', length=seq_length)
    coords = seq.numpy()
    
    # Calculate step sizes between consecutive bricks
    for j in range(1, seq_length):
        step = coords[j] - coords[j-1]
        step_sizes.append(np.linalg.norm(step))
        z_increments.append(step[2])  # Z component
        
        # Calculate 2D direction in XY plane
        if np.linalg.norm(step[:2]) > 0:
            xy_direction = step[:2] / np.linalg.norm(step[:2])
            directions.append(xy_direction)
    
    # Visualize this random walk
    if i < 3:  # Only visualize first 3 for clarity
        visualize_sequence(
            coords,
            show_indices=True,
            title=f"Random Walk Sample {i+1}",
            save_path=os.path.join(output_dir, f'random_walk_sample_{i+1}.png')
        )

# Plot histograms of step properties
plt.figure(figsize=(15, 5))

# Step sizes histogram
plt.subplot(131)
plt.hist(step_sizes, bins=20)
plt.title('Step Sizes')
plt.xlabel('Euclidean Distance')
plt.ylabel('Frequency')

# Z increments histogram
plt.subplot(132)
plt.hist(z_increments, bins=20)
plt.title('Z-Axis Increments')
plt.xlabel('Z Change')
plt.ylabel('Frequency')

# 2D Direction scatter plot
directions = np.array(directions)
plt.subplot(133)
plt.scatter(directions[:, 0], directions[:, 1], alpha=0.5)
plt.title('XY-Plane Directions')
plt.xlabel('X Component')
plt.ylabel('Y Component')
plt.axis('equal')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'random_walk_statistics.png'))

# 2. Load the original model and the specialized random walk model
print("\nComparing models on random walk prediction...")

# Try to load specialized model
random_walk_model_path = './checkpoints/random_walk_model/best_model.pt'
original_model_path = './checkpoints/best_model.pt'

# Create test data - generate several random walks
test_sequences = [generate_brick_sequence('random_walk', length=10) for _ in range(5)]
test_batch = torch.stack(test_sequences, dim=0)
inputs = test_batch[:, :-1, :].to(device)
targets = test_batch[:, 1:, :].to(device)

# Function to load a model from checkpoint
def load_model(checkpoint_path, model_class=LegoTransformer):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Determine if this is a state_dict or full checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Initialize model architecture
        if checkpoint_path == random_walk_model_path:
            model = model_class(d_model=64, nhead=8, num_layers=3, dim_feedforward=128)
        else:
            model = model_class()
            
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model from {checkpoint_path}: {e}")
        return None

# Load models
original_model = load_model(original_model_path)
random_walk_model = load_model(random_walk_model_path)

# Compare performance if both models are available
if original_model and random_walk_model:
    # Test both models
    loss_fn = torch.nn.MSELoss()
    
    with torch.no_grad():
        # Original model prediction
        original_outputs = original_model(inputs)
        original_loss = loss_fn(original_outputs, targets).item()
        
        # Random walk model prediction
        random_walk_outputs = random_walk_model(inputs)
        random_walk_loss = loss_fn(random_walk_outputs, targets).item()
        
    print(f"Original Model Loss: {original_loss:.6f}")
    print(f"Random Walk Model Loss: {random_walk_loss:.6f}")
    
    # Visualize predictions from both models for the first sequence
    plt.figure(figsize=(15, 5))
    
    # Ground truth
    first_sequence = test_sequences[0].numpy()
    ax1 = plt.subplot(131, projection='3d')
    ax1.scatter(first_sequence[:, 0], first_sequence[:, 1], first_sequence[:, 2], 
              c=range(len(first_sequence)), cmap='viridis')
    ax1.set_title('Ground Truth')
    
    # Original model
    with torch.no_grad():
        start_brick = test_sequences[0][:1].unsqueeze(0).to(device)
        original_seq = generate_sequence(original_model, start_brick, len(first_sequence)-1, device)
        original_seq = original_seq.squeeze(0).cpu().numpy()
    
    ax2 = plt.subplot(132, projection='3d')
    ax2.scatter(original_seq[:, 0], original_seq[:, 1], original_seq[:, 2], 
              c=range(len(original_seq)), cmap='viridis')
    ax2.set_title('Original Model')
    
    # Random walk model
    with torch.no_grad():
        start_brick = test_sequences[0][:1].unsqueeze(0).to(device)
        random_walk_seq = generate_sequence(random_walk_model, start_brick, len(first_sequence)-1, device)
        random_walk_seq = random_walk_seq.squeeze(0).cpu().numpy()
    
    ax3 = plt.subplot(133, projection='3d')
    ax3.scatter(random_walk_seq[:, 0], random_walk_seq[:, 1], random_walk_seq[:, 2], 
              c=range(len(random_walk_seq)), cmap='viridis')
    ax3.set_title('Random Walk Model')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
else:
    print("One or both models could not be loaded for comparison")

# 3. Analyze why random walks are hard to predict
print("\nWhy random walks are difficult to predict:")
print("1. Random walks contain actual randomness by design")
print("2. The next position depends on random factors that aren't predictable from history")
print("3. A good model might learn the distribution of steps rather than exact positions")
print("4. MSE loss may not be ideal for truly random processes")

# Suggest a different approach
print("\nPossible solutions:")
print("1. Instead of coordinate regression, predict step size and direction distributions")
print("2. Use a different loss function that accounts for positional uncertainty")
print("3. Generate multiple possible outcomes instead of a single prediction")
print("4. Consider past K steps to capture short-term patterns in the randomness")

print("\nAnalysis complete. Results saved to", output_dir) 