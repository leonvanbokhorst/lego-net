#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for LEGO-Net components
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from lego_net.data.data_generator import generate_brick_sequence, visualize_sequence
from lego_net.model.transformer import LegoTransformer, count_parameters


def test_data_generator():
    """Test the data generation module"""
    print("\n=== Testing Data Generator ===")
    
    patterns = ["stack", "row", "stair", "random_walk"]
    
    plt.figure(figsize=(15, 10))
    
    for i, pattern in enumerate(patterns):
        print(f"Generating {pattern} pattern...")
        
        # Generate sequence
        seq = generate_brick_sequence(pattern, length=8)
        print(f"  Shape: {seq.shape}")
        print(f"  First few coordinates: {seq[:3].tolist()}")
        
        # Plot in subplot
        plt.subplot(2, 2, i+1)
        ax = plt.gca()
        ax.scatter(seq[:, 0], seq[:, 2], c=range(len(seq)), cmap='viridis', s=100)
        
        for j, (x, y, z) in enumerate(seq):
            ax.text(x, z, f" {j}", fontsize=10, color='red')
        
        # Connect points with lines
        ax.plot(seq[:, 0], seq[:, 2], 'k--', alpha=0.4)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title(f'Pattern: {pattern}')
        
        # Equal aspect ratio
        max_range = max(np.ptp(seq[:, 0].numpy()), np.ptp(seq[:, 2].numpy()))
        ax.set_aspect('equal')
        
    plt.tight_layout()
    plt.savefig('pattern_test.png')
    plt.show()
    
    # Test 3D visualization
    print("\nTesting 3D visualization...")
    for pattern in patterns:
        seq = generate_brick_sequence(pattern, length=8)
        visualize_sequence(seq, title=f"Pattern: {pattern}")


def test_model():
    """Test the transformer model"""
    print("\n=== Testing Transformer Model ===")
    
    # Create a small model
    model = LegoTransformer(d_model=32, nhead=4, num_layers=2)
    print(f"Created model with {count_parameters(model):,} trainable parameters")
    
    # Test with a small batch
    batch_size, seq_len = 2, 5
    x = torch.randn(batch_size, seq_len, 3)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0, 0, :].tolist()}")
    
    # Check autoregressive generation (one step)
    model.eval()
    with torch.no_grad():
        # Start with just one brick
        start_seq = x[:, :1, :]
        print(f"Start sequence shape: {start_seq.shape}")
        
        # Get prediction for the next brick
        pred = model(start_seq)
        print(f"Prediction shape: {pred.shape}")
        
        # Extract the last prediction
        next_brick = pred[:, -1:, :]  # Shape [batch_size, 1, 3]
        print(f"Next brick shape: {next_brick.shape}")
        
        # Append to sequence
        extended_seq = torch.cat([start_seq, next_brick], dim=1)
        print(f"Extended sequence shape: {extended_seq.shape}")


def test_forward_pass_toy_example():
    """Test the model with a simple toy example"""
    print("\n=== Testing Forward Pass with Toy Example ===")
    
    # Create model
    model = LegoTransformer(d_model=32, nhead=4, num_layers=2)
    model.eval()
    
    # Create a stack pattern sequence
    seq = generate_brick_sequence("stack", length=4)
    inputs = seq[:-1].unsqueeze(0)  # shape [1, 3, 3]
    targets = seq[1:].unsqueeze(0)  # shape [1, 3, 3]
    
    print(f"Input sequence: {inputs.squeeze().tolist()}")
    print(f"Target sequence: {targets.squeeze().tolist()}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)
    
    print(f"Output sequence: {outputs.squeeze().tolist()}")
    
    # Calculate error
    mse = torch.mean((outputs - targets) ** 2).item()
    print(f"MSE: {mse:.6f}")
    
    # Improvement after training would reduce this MSE


if __name__ == "__main__":
    # Run tests
    test_data_generator()
    test_model()
    test_forward_pass_toy_example() 