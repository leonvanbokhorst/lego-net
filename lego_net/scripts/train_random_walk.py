#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train a model specifically on random_walk patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Update imports to use full paths from root package
from lego_net.model.transformer import LegoTransformer
from lego_net.data.data_generator import generate_brick_sequence
from lego_net.utils.training import generate_sequence
from lego_net.data.data_generator import visualize_sequence

# Define the custom data generator for random_walk patterns
def custom_generate_batch(batch_size, seq_length):
    # Generate sequences specifically with random_walk pattern
    sequences = [generate_brick_sequence('random_walk', seq_length) for _ in range(batch_size)]
    batch = torch.stack(sequences, dim=0)
    # Split into input and target
    inputs = batch[:, :-1, :]
    targets = batch[:, 1:, :]
    return inputs, targets

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
model = LegoTransformer(
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

# Create optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Create checkpoint directory
checkpoint_dir = './checkpoints/random_walk_model'
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop
training_losses = []
eval_losses = []

print("Training random_walk-specific model...")
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    
    # Training batches
    num_batches = 30
    for _ in tqdm(range(num_batches), desc=f"Epoch {epoch}/{epochs}"):
        # Generate batch of random_walk patterns
        inputs, targets = custom_generate_batch(batch_size, seq_length)
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / num_batches
    training_losses.append(avg_train_loss)
    
    # Evaluation
    if epoch % 10 == 0 or epoch == epochs:
        model.eval()
        eval_loss = 0
        num_eval_batches = 10
        
        with torch.no_grad():
            for _ in range(num_eval_batches):
                test_inputs, test_targets = custom_generate_batch(batch_size, seq_length)
                test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
                test_outputs = model(test_inputs)
                test_loss = loss_fn(test_outputs, test_targets).item()
                eval_loss += test_loss
        
        avg_eval_loss = eval_loss / num_eval_batches
        eval_losses.append(avg_eval_loss)
        
        print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.6f}, Eval Loss: {avg_eval_loss:.6f}")
        
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
            print(f"Saved best model at epoch {epoch}")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), training_losses, label='Training Loss')
plt.plot([i * 10 for i in range(len(eval_losses))], eval_losses, 'o-', label='Evaluation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training History for Random Walk Model')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(checkpoint_dir, 'training_history.png'))

# Generate and visualize sequences
print("\nGenerating sequences with the trained model...")
model.eval()

# Create output directory
output_dir = './output/random_walk_model'
os.makedirs(output_dir, exist_ok=True)

# Generate a few examples with different starting positions
for i in range(3):
    # Create a random starting position
    start_seq = generate_brick_sequence('random_walk', length=1)
    start_brick = start_seq.unsqueeze(0).to(device)  # Shape [1, 1, 3]
    
    # Generate sequence
    gen_length = 15
    generated = generate_sequence(model, start_brick, gen_length, device)
    
    # Convert to numpy for visualization
    seq_coords = generated.squeeze(0).cpu().numpy()
    
    # Visualize
    title = f"Generated Random Walk Pattern {i+1}"
    save_path = os.path.join(output_dir, f'generated_random_walk_{i+1}.png')
    
    visualize_sequence(
        seq_coords,
        show_indices=True,
        title=title,
        save_path=save_path
    )
    
    print(f"Generated sequence {i+1} saved to {save_path}")

print("Done! Model saved to", checkpoint_dir) 