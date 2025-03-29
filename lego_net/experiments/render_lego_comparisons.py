#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to render LEGO brick visualizations for our model comparisons
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from lego_net.model.transformer import LegoTransformer
from lego_net.model.step_predictor import StepPredictorTransformer
from lego_net.data.data_generator import generate_brick_sequence
from lego_net.utils.training import generate_sequence as generate_sequence_original
from lego_net.visualization.lego_renderer import render_lego_sequence, visualize_model_comparison

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory
output_dir = './output/lego_visualizations'
os.makedirs(output_dir, exist_ok=True)
# Create subdirectories
comparisons_dir = os.path.join(output_dir, 'comparisons')
models_dir = os.path.join(output_dir, 'models')
single_dir = os.path.join(output_dir, 'single')
os.makedirs(comparisons_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(single_dir, exist_ok=True)


def load_model(model_path, model_class):
    """Load a model from a checkpoint file."""
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


def generate_lego_comparisons():
    """Generate LEGO brick visualizations comparing all models."""
    # Define model paths
    model_paths = {
        'original': './checkpoints/best_model.pt',
        'random_walk': './checkpoints/random_walk_model/best_model.pt',
        'step_predictor': './checkpoints/step_predictor_model/best_model.pt'
    }
    
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
        return
    
    # Generate and visualize for each pattern type
    patterns = ['stack', 'row', 'stair', 'random_walk']
    
    for pattern in patterns:
        print(f"\nVisualizing {pattern} pattern with LEGO bricks...")
        
        # Generate a starting sequence
        seed_length = 5
        seed_seq = generate_brick_sequence(pattern, length=seed_length)
        start_brick = seed_seq.unsqueeze(0).to(device)  # [1, seed_length, 3]
        
        # Generate sequences with each model
        sequences = {}
        
        for name in available_models:
            model = models[name]
            
            with torch.no_grad():
                if name == 'step_predictor':
                    # Step predictor model has its own generation method
                    generated = model.generate_sequence(start_brick, 10)
                else:
                    # Original and random_walk models
                    # Use only the first 2 bricks as seed to make it more challenging
                    generated = generate_sequence_original(model, start_brick[:, :2, :], 13, device)
                
                sequences[name] = generated.squeeze(0).cpu().numpy()
        
        # Create individual LEGO visualizations for each model
        for name, coords in sequences.items():
            title = f"{name.capitalize()} Model - {pattern.capitalize()} Pattern"
            output_path = os.path.join(models_dir, f"lego_{pattern}_{name}.png")
            
            # Render LEGO sequence
            render_lego_sequence(coords, output_path, title)
            print(f"  Created {name} model visualization for {pattern} pattern")
        
        # Create a comparison visualization
        visualize_model_comparison(sequences, pattern, comparisons_dir)
        print(f"  Created comparison visualization for {pattern} pattern")
    
    print("\nAll LEGO visualizations saved to", output_dir)


def render_single_lego_sequence(pattern='stack', model_name='step_predictor', length=15):
    """Render a single LEGO sequence for the given pattern and model."""
    # Load the specified model
    if model_name == 'step_predictor':
        model_path = './checkpoints/step_predictor_model/best_model.pt'
        model = load_model(model_path, StepPredictorTransformer)
    elif model_name == 'random_walk':
        model_path = './checkpoints/random_walk_model/best_model.pt'
        model = load_model(model_path, LegoTransformer)
    else:
        model_path = './checkpoints/best_model.pt'
        model = load_model(model_path, LegoTransformer)
    
    if model is None:
        print(f"Could not load {model_name} model")
        return
    
    # Generate a seed sequence
    seed_length = 5
    seed_seq = generate_brick_sequence(pattern, length=seed_length)
    start_brick = seed_seq.unsqueeze(0).to(device)
    
    # Generate the sequence
    with torch.no_grad():
        if model_name == 'step_predictor':
            generated = model.generate_sequence(start_brick, length)
        else:
            generated = generate_sequence_original(model, start_brick[:, :2, :], length+3, device)
    
    coords = generated.squeeze(0).cpu().numpy()
    
    # Render the LEGO sequence
    title = f"{model_name.capitalize()} Model - {pattern.capitalize()} Pattern"
    output_path = os.path.join(single_dir, f"single_lego_{pattern}_{model_name}.png")
    
    fig, ax = render_lego_sequence(coords, output_path, title)
    plt.show()
    
    print(f"Rendered single {pattern} pattern using {model_name} model")
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    # Generate all comparisons
    generate_lego_comparisons()
    
    # Or render a single sequence (uncomment to use)
    # render_single_lego_sequence('stack', 'step_predictor', 15) 