# LEGO Brick Placement Transformer Models

This project implements and compares different neural network architectures for predicting LEGO brick placements in 3D space.

## Project Structure

```
.
├── checkpoints/             # Model checkpoints
│   ├── best_model.pt        # Original model
│   ├── random_walk_model/   # Random walk specific model
│   └── step_predictor_model/# Step-based predictor model
├── lego_net/                # Main package
│   ├── data/                # Data generation utilities
│   ├── experiments/         # Analysis and comparison scripts
│   ├── model/               # Model architectures 
│   ├── scripts/             # Training scripts
│   ├── utils/               # Utility functions
│   ├── visualization/       # Visualization utilities
│   ├── __init__.py          
│   ├── main.py              # Main entry point
│   └── test.py              # Test script
├── output/                  # Generated outputs and visualizations
├── .venv/                   # Virtual environment
├── README.md                # This file
├── run.py                   # Unified runner script
└── requirements.txt         # Project dependencies
```

## Models

We implement and compare three different neural network architectures:

1. **Original Transformer** - A basic transformer decoder model that predicts the absolute coordinates of the next brick
2. **Random Walk Specific Model** - A specialized version of the original model trained specifically on random walk patterns
3. **Step-Based Predictor** - An advanced model that predicts relative step vectors between bricks rather than absolute positions

## Key Findings

The step-based predictor significantly outperforms the other models across all pattern types:

| Pattern Type | Original Model | Random Walk Model | Step Predictor |
|--------------|----------------|-------------------|----------------|
| Random Walk  | 0.196552       | 0.128793          | **0.114754**   |
| Stack        | 0.898832       | 0.417523          | **0.001951**   | 
| Row          | 0.917026       | 3.756550          | **0.013034**   |
| Stair        | 2.268482       | 6.004579          | **0.007892**   |

Values represent Mean Squared Error (lower is better).

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lego-net.git
cd lego-net

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Using the run.py script (recommended)

The `run.py` script provides a unified interface to run all available models and experiments:

```bash
# Show available commands
./run.py --help

# Train the original model
./run.py original

# Train the random walk model
./run.py random_walk

# Train the step-based predictor model
./run.py step_predictor

# Analyze random walk patterns
./run.py analyze

# Compare all models
./run.py compare

# Run all models and experiments with 50 epochs each
./run.py all --epochs 50
```

### Alternative: Run individual modules directly

```bash
# Train the original model
python -m lego_net.main

# Train the random walk specific model
python -m lego_net.scripts.train_random_walk

# Train the step-based predictor model
python -m lego_net.scripts.train_step_predictor

# Compare all models
python -m lego_net.experiments.compare_models
```

## Visualizations

The `output/model_comparison` directory contains visualizations comparing how each model predicts different LEGO brick patterns.

## Conclusion

The step-based transformer model represents a significant improvement for LEGO brick sequence prediction. By focusing on the relations between bricks rather than their absolute positions, the model better captures the underlying patterns in LEGO constructions.

This approach could be extended to more complex LEGO models with varying brick types and orientations, potentially enabling automated LEGO construction planning and interactive building assistance. 