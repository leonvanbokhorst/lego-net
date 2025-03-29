#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for LEGO-Net: Train and test a transformer model to generate LEGO brick sequences
"""

import torch
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
from time import time

from lego_net.model.transformer import LegoTransformer, count_parameters
from lego_net.data.data_generator import generate_batch, generate_brick_sequence, visualize_sequence
from lego_net.utils.training import train_model, generate_sequence, plot_training_history, load_model


def train_new_model(args):
    """
    Train a new LEGO-Net model
    """
    print("\n=== Training new LEGO-Net model ===")
    
    # Create model
    model = LegoTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )
    
    # Print model info
    print(f"Created model with {count_parameters(model):,} trainable parameters")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Train the model
    history = train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        batches_per_epoch=args.batches_per_epoch,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        save_dir=args.checkpoint_dir
    )
    
    # Plot training history
    plot_training_history(history, os.path.join(args.checkpoint_dir, 'training_history.png'))
    
    return model


def generate_and_visualize(model, device, args):
    """
    Generate and visualize LEGO sequences using the trained model
    """
    print("\n=== Generating LEGO sequences ===")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate sequences for each pattern
    patterns = ["stack", "row", "stair", "random_walk", None]
    pattern_names = ["stack", "row", "stair", "random_walk", "custom"]
    
    for pattern, name in zip(patterns, pattern_names):
        print(f"\nGenerating and visualizing {name} pattern:")
        
        # For custom pattern, use a specified starting position
        if pattern is None:
            start_pos = (1.0, 0.5, 0.0)  # Example custom starting position
            start_brick = torch.tensor([[[*start_pos]]], dtype=torch.float32)
        else:
            # Use the first brick from a generated sequence
            seed_seq = generate_brick_sequence(pattern, length=1)
            start_brick = seed_seq.unsqueeze(0)  # Shape [1, 1, 3]
        
        # Generate sequence
        print(f"  Starting with brick at {start_brick.squeeze().tolist()}")
        generated = generate_sequence(
            model, start_brick, args.gen_length, device
        )
        
        # Get as numpy array for visualization
        seq_coords = generated.squeeze(0).cpu().numpy()
        
        # Visualize
        print(f"  Generated {len(seq_coords)} bricks")
        title = f"Generated {name.capitalize()} Pattern"
        save_path = os.path.join(args.output_dir, f'generated_{name}.png')
        
        visualize_sequence(
            seq_coords, 
            show_indices=True,
            title=title,
            save_path=save_path
        )
        
        # Print coordinates
        print(f"  Coordinates: {seq_coords.tolist()}")
        
    print(f"\nAll visualizations saved to {args.output_dir}")


def evaluate_patterns(model, device, args):
    """
    Evaluate the model's accuracy on different patterns
    """
    print("\n=== Evaluating pattern accuracy ===")
    
    # Define patterns to test
    patterns = ["stack", "row", "stair", "random_walk"]
    results = {}
    
    for pattern in patterns:
        print(f"\nEvaluating {pattern} pattern:")
        
        # Create test batches with known patterns
        inputs, targets = generate_batch(
            batch_size=args.batch_size, 
            seq_length=args.seq_length, 
            pattern=pattern
        )
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        
        # Calculate MSE
        mse = torch.mean((outputs - targets) ** 2).item()
        print(f"  MSE: {mse:.6f}")
        
        # Calculate directional accuracy (how well the model predicts the direction)
        # Compute the direction vectors between consecutive bricks in both target and prediction
        target_dirs = targets[:, 1:, :] - targets[:, :-1, :]
        output_dirs = outputs[:, 1:, :] - outputs[:, :-1, :]
        
        # Normalize to unit vectors
        target_dirs_norm = target_dirs / (torch.norm(target_dirs, dim=2, keepdim=True) + 1e-6)
        output_dirs_norm = output_dirs / (torch.norm(output_dirs, dim=2, keepdim=True) + 1e-6)
        
        # Compute cosine similarity
        cos_sim = torch.sum(target_dirs_norm * output_dirs_norm, dim=2)
        dir_accuracy = torch.mean(cos_sim).item()
        
        print(f"  Directional accuracy: {dir_accuracy:.6f}")
        
        results[pattern] = {
            'mse': mse,
            'dir_accuracy': dir_accuracy
        }
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot MSE
    plt.subplot(1, 2, 1)
    plt.bar(results.keys(), [r['mse'] for r in results.values()])
    plt.title('MSE by Pattern Type')
    plt.ylabel('Mean Squared Error')
    plt.ylim(bottom=0)
    
    # Plot directional accuracy
    plt.subplot(1, 2, 2)
    plt.bar(results.keys(), [r['dir_accuracy'] for r in results.values()])
    plt.title('Directional Accuracy by Pattern Type')
    plt.ylabel('Cosine Similarity')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    eval_path = os.path.join(args.output_dir, 'pattern_evaluation.png')
    plt.savefig(eval_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Evaluation results saved to {eval_path}")


def main():
    parser = argparse.ArgumentParser(description="LEGO-Net: Transform-based LEGO brick sequence generation")
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=32, help='Dimension of model embeddings')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=64, help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_length', type=int, default=6, help='Sequence length')
    parser.add_argument('--batches_per_epoch', type=int, default=50, help='Batches per epoch')
    parser.add_argument('--eval_batches', type=int, default=10, help='Evaluation batches')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    # Generation parameters
    parser.add_argument('--gen_length', type=int, default=10, help='Length of generated sequences')
    
    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--load_model', type=str, default=None, help='Path to model checkpoint to load')
    
    # Action flags
    parser.add_argument('--skip_training', action='store_true', help='Skip training and load model')
    parser.add_argument('--skip_generation', action='store_true', help='Skip sequence generation')
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip pattern evaluation')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train or load model
    if args.skip_training and args.load_model:
        print(f"\nLoading model from {args.load_model}")
        model, _ = load_model(args.load_model, device=device)
        print(f"Loaded model with {count_parameters(model):,} parameters")
    elif args.skip_training:
        # Try to load the latest model
        best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            print(f"\nLoading best model from {best_model_path}")
            model, _ = load_model(best_model_path, device=device)
            print(f"Loaded model with {count_parameters(model):,} parameters")
        else:
            print("\nNo model found to load. Training new model...")
            model = train_new_model(args)
    else:
        # Train new model
        model = train_new_model(args)
    
    # Make sure model is on the correct device
    model = model.to(device)
    
    # Generate sequences
    if not args.skip_generation:
        generate_and_visualize(model, device, args)
    
    # Evaluate on different patterns
    if not args.skip_evaluation:
        evaluate_patterns(model, device, args)


if __name__ == "__main__":
    main() 