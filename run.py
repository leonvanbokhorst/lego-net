#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run script for LEGO-Net models and experiments.

This script provides a simple interface to run different models and experiments.
"""

import argparse
import os
import sys
import subprocess

def print_header(text):
    """Print a stylized header."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)

def run_module(module_name, args=None):
    """Run a Python module with the specified args."""
    cmd = [sys.executable, "-m", module_name]
    if args:
        cmd.extend(args)
    
    print_header(f"Running {module_name}")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Run LEGO-Net models and experiments")
    
    # Define the available commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Original model
    original_parser = subparsers.add_parser("original", help="Train the original model")
    original_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    original_parser.add_argument("--skip_training", action="store_true", help="Skip training and load model")
    
    # Random walk model
    random_walk_parser = subparsers.add_parser("random_walk", help="Train the random walk model")
    
    # Step predictor model
    step_predictor_parser = subparsers.add_parser("step_predictor", help="Train the step predictor model")
    
    # Analyze random walk
    analyze_parser = subparsers.add_parser("analyze", help="Analyze random walk patterns")
    
    # Compare models
    compare_parser = subparsers.add_parser("compare", help="Compare all models")
    
    # LEGO visualization
    lego_parser = subparsers.add_parser("lego", help="Generate LEGO brick visualizations")
    lego_parser.add_argument("--pattern", type=str, default=None, 
                            choices=["stack", "row", "stair", "random_walk"], 
                            help="Pattern to visualize (all patterns if not specified)")
    lego_parser.add_argument("--model", type=str, default=None,
                            choices=["original", "random_walk", "step_predictor"],
                            help="Model to use (all models if not specified)")
    
    # Single LEGO visualization
    single_lego_parser = subparsers.add_parser("single_lego", help="Generate a single LEGO brick visualization")
    single_lego_parser.add_argument("--pattern", type=str, required=True, 
                                   choices=["stack", "row", "stair", "random_walk"], 
                                   help="Pattern to visualize")
    single_lego_parser.add_argument("--model", type=str, required=True,
                                   choices=["original", "random_walk", "step_predictor"],
                                   help="Model to use")
    single_lego_parser.add_argument("--length", type=int, default=15,
                                   help="Number of bricks to generate")
    
    # All in one (run everything)
    all_parser = subparsers.add_parser("all", help="Run all models and experiments")
    all_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for each model")
    
    args = parser.parse_args()
    
    if args.command == "original":
        # Run the original model
        cmd_args = []
        if args.epochs != 100:
            cmd_args.extend(["--epochs", str(args.epochs)])
        if args.skip_training:
            cmd_args.append("--skip_training")
        run_module("lego_net.main", cmd_args)
        
    elif args.command == "random_walk":
        # Run the random walk model
        run_module("lego_net.scripts.train_random_walk")
        
    elif args.command == "step_predictor":
        # Run the step predictor model
        run_module("lego_net.scripts.train_step_predictor")
        
    elif args.command == "analyze":
        # Run the analysis
        run_module("lego_net.experiments.analyze_random_walk")
        
    elif args.command == "compare":
        # Run the comparison
        run_module("lego_net.experiments.compare_models")
        
    elif args.command == "lego":
        # Run the LEGO visualization
        run_module("lego_net.experiments.render_lego_comparisons")
        
    elif args.command == "single_lego":
        # Create an inline Python script to call the single LEGO function with parameters
        code = f"""
from lego_net.experiments.render_lego_comparisons import render_single_lego_sequence
render_single_lego_sequence('{args.pattern}', '{args.model}', {args.length})
"""
        cmd = [sys.executable, "-c", code]
        print_header(f"Rendering single LEGO sequence: {args.pattern} pattern with {args.model} model")
        subprocess.run(cmd)
        
    elif args.command == "all":
        # Run everything
        
        # Train original model
        run_module("lego_net.main", ["--epochs", str(args.epochs)])
        
        # Train random walk model
        # Modify the source file to set the epochs
        with open("lego_net/scripts/train_random_walk.py", "r") as f:
            content = f.read()
        content = content.replace("epochs = 100", f"epochs = {args.epochs}")
        with open("lego_net/scripts/train_random_walk.py", "w") as f:
            f.write(content)
        run_module("lego_net.scripts.train_random_walk")
        
        # Train step predictor model
        # Modify the source file to set the epochs
        with open("lego_net/scripts/train_step_predictor.py", "r") as f:
            content = f.read()
        content = content.replace("epochs = 100", f"epochs = {args.epochs}")
        with open("lego_net/scripts/train_step_predictor.py", "w") as f:
            f.write(content)
        run_module("lego_net.scripts.train_step_predictor")
        
        # Run analysis
        run_module("lego_net.experiments.analyze_random_walk")
        
        # Run comparison
        run_module("lego_net.experiments.compare_models")
        
        # Generate LEGO visualizations
        run_module("lego_net.experiments.render_lego_comparisons")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 