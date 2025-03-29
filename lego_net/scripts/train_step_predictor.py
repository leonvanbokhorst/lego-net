#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train the step-based predictor model for random walks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Update imports to use full paths from root package
from lego_net.model.step_predictor import StepPredictorTransformer, count_parameters
from lego_net.data.data_generator import generate_brick_sequence, visualize_sequence

# Define the custom data generator for random_walk patterns
def generate_batch(batch_size, seq_length, pattern=None):
    # Generate sequences
    sequences = [generate_brick_sequence(pattern, seq_length) for _ in range(batch_size)]
    batch = torch.stack(sequences, dim=0)
    return batch

# Set hyperparameters
d_model = 64
nhead = 8
num_layers = 3
dim_feedforward = 128
epochs = 50
batch_size = 64
learning_rate = 0.001
seq_length = 10

# Create model with enhanced capacity
model = StepPredictorTransformer(
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")
print(f"Model has {count_parameters(model):,} trainable parameters")

# Create optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Create checkpoint directory
checkpoint_dir = './checkpoints/step_predictor_model'
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop
training_losses = []
eval_losses = []
step_losses = []
coord_losses = []

# Mix of patterns for more robust training
patterns = ['random_walk', 'stack', 'row', 'stair', None]

print("Training step-based predictor model...")
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    epoch_step_loss = 0
    epoch_coord_loss = 0
    
    # Training batches
    num_batches = 30
    for _ in tqdm(range(num_batches), desc=f"Epoch {epoch}/{epochs}"):
        # Generate a mixed batch of patterns
        pattern = np.random.choice(patterns)
        coords_batch = generate_batch(batch_size, seq_length, pattern)
        coords_batch = coords_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_steps, pred_coords = model(coords_batch)
        
        # Calculate target steps
        true_steps = coords_batch[:, 1:, :] - coords_batch[:, :-1, :]
        
        # Calculate losses
        step_loss = loss_fn(pred_steps, true_steps)
        coord_loss = loss_fn(pred_coords[:, 1:, :], coords_batch[:, 1:, :])
        
        # Combined loss with more weight on step prediction
        loss = 0.7 * step_loss + 0.3 * coord_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_step_loss += step_loss.item()
        epoch_coord_loss += coord_loss.item()
    
    avg_train_loss = epoch_loss / num_batches
    avg_step_loss = epoch_step_loss / num_batches
    avg_coord_loss = epoch_coord_loss / num_batches
    
    training_losses.append(avg_train_loss)
    step_losses.append(avg_step_loss)
    coord_losses.append(avg_coord_loss)
    
    # Evaluation
    if epoch % 10 == 0 or epoch == epochs:
        model.eval()
        eval_loss = 0
        eval_step_loss = 0
        eval_coord_loss = 0
        num_eval_batches = 10
        
        with torch.no_grad():
            for _ in range(num_eval_batches):
                # Generate random walk patterns for evaluation
                pattern = 'random_walk'
                coords_batch = generate_batch(batch_size, seq_length, pattern)
                coords_batch = coords_batch.to(device)
                
                # Forward pass
                pred_steps, pred_coords = model(coords_batch)
                
                # Calculate target steps
                true_steps = coords_batch[:, 1:, :] - coords_batch[:, :-1, :]
                
                # Calculate losses
                step_loss = loss_fn(pred_steps, true_steps)
                coord_loss = loss_fn(pred_coords[:, 1:, :], coords_batch[:, 1:, :])
                
                # Combined loss
                loss = 0.7 * step_loss + 0.3 * coord_loss
                
                eval_loss += loss.item()
                eval_step_loss += step_loss.item()
                eval_coord_loss += coord_loss.item()
        
        avg_eval_loss = eval_loss / num_eval_batches
        eval_losses.append(avg_eval_loss)
        
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f} (Step: {avg_step_loss:.6f}, Coord: {avg_coord_loss:.6f})")
        print(f"  Eval Loss: {avg_eval_loss:.6f} (Step: {eval_step_loss/num_eval_batches:.6f}, Coord: {eval_coord_loss/num_eval_batches:.6f})")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'eval_loss': avg_eval_loss
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt'))
        
        # Save best model
        if epoch == 10 or avg_eval_loss < min(eval_losses[:-1], default=float('inf')):
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"  Saved best model at epoch {epoch}")

# Plot training history
plt.figure(figsize=(15, 5))

# Combined loss
plt.subplot(131)
plt.plot(range(1, epochs + 1), training_losses, label='Training Loss')
plt.plot([i * 10 for i in range(len(eval_losses))], eval_losses, 'o-', label='Evaluation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (Combined)')
plt.title('Combined Loss')
plt.legend()
plt.grid(True)

# Step loss
plt.subplot(132)
plt.plot(range(1, epochs + 1), step_losses, label='Step Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (Steps)')
plt.title('Step Vector Loss')
plt.grid(True)

# Coordinate loss
plt.subplot(133)
plt.plot(range(1, epochs + 1), coord_losses, label='Coordinate Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (Coordinates)')
plt.title('Coordinate Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, 'training_history.png'))

# Generate and visualize sequences
print("\nGenerating sequences with the trained model...")
model.eval()

# Create output directory
output_dir = './output/step_predictor_model'
os.makedirs(output_dir, exist_ok=True)

# Function to generate sequences with the model
def generate_and_visualize(pattern, idx, num_to_generate=5):
    # Generate a starting sequence
    start_seq = generate_brick_sequence(pattern, length=5)
    start_seq_batch = start_seq.unsqueeze(0).to(device)  # [1, 5, 3]
    
    # Generate additional steps
    gen_length = 15
    generated = model.generate_sequence(start_seq_batch, gen_length)
    
    # Convert to numpy for visualization
    start_coords = start_seq.numpy()
    gen_coords = generated.squeeze(0).cpu().numpy()
    
    # Create a plot with both original and extended sequence
    fig = plt.figure(figsize=(15, 6))
    
    # Plot original sequence
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(start_coords[:, 0], start_coords[:, 1], start_coords[:, 2],
              c=range(len(start_coords)), cmap='viridis', s=100)
    
    # Connect the dots
    ax1.plot(start_coords[:, 0], start_coords[:, 1], start_coords[:, 2], 'k--', alpha=0.4)
    
    # Add labels
    for i, (x, y, z) in enumerate(start_coords):
        ax1.text(x, y, z, f" {i}", fontsize=10, color='red')
        
    ax1.set_title(f'Initial Sequence ({pattern})')
    
    # Plot generated sequence
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(gen_coords[:, 0], gen_coords[:, 1], gen_coords[:, 2],
              c=range(len(gen_coords)), cmap='viridis', s=100)
    
    # Connect the dots
    ax2.plot(gen_coords[:, 0], gen_coords[:, 1], gen_coords[:, 2], 'k--', alpha=0.4)
    
    # Add labels
    for i, (x, y, z) in enumerate(gen_coords):
        ax2.text(x, y, z, f" {i}", fontsize=10, color='red')
        
    ax2.set_title(f'Generated Sequence (Extended by {gen_length} steps)')
    
    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'generated_{pattern}_{idx+1}.png')
    plt.savefig(save_path)
    plt.close(fig)
    
    return gen_coords, save_path

# Generate for each pattern type
for i, pattern in enumerate(['random_walk', 'stack', 'row', 'stair']):
    for j in range(2):  # Generate 2 examples of each
        coords, path = generate_and_visualize(pattern, j)
        print(f"Generated {pattern} pattern {j+1}, saved to {path}")

print("Done! Model saved to", checkpoint_dir) 