# LEGO-Net: Transformer for LEGO Brick Placement

A transformer-based neural network that learns to generate sequences of LEGO brick placements. 🧱✨

## Overview

This project demonstrates a small transformer-style neural network that learns to generate sequences of 2×4 LEGO brick placements. The model uses a Transformer decoder architecture to predict the next brick's position given previous bricks, treating brick positions as a sequence.

Key features:
- Synthetic data generation with different brick placement patterns
- Lightweight transformer model with self-attention
- Regression-based coordinate prediction
- Autoregressive sequence generation
- 3D visualization of generated structures

## Project Structure

```
lego_net/
├── data/               # Data generation modules
├── model/              # Transformer model implementation
├── utils/              # Training and utility functions
├── visualization/      # Visualization tools
├── main.py             # Main script for training and generation
└── test.py             # Quick tests for components
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/lego-net.git
cd lego-net
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Test

To test the basic functionality:

```bash
python -m lego_net.test
```

This will:
- Generate and visualize sample brick patterns
- Test the transformer model's forward pass
- Run a toy example with untrained model

### Training a Model

```bash
python -m lego_net.main
```

This will:
1. Train a new transformer model on synthetic LEGO patterns
2. Save checkpoints to the `./checkpoints` directory
3. Generate and visualize new sequences
4. Evaluate the model on different pattern types

### Command-line Options

You can customize training and generation with various options:

```bash
python -m lego_net.main --epochs 200 --d_model 64 --num_layers 3
```

For all available options:

```bash
python -m lego_net.main --help
```

### Generating Sequences Only

To skip training and just generate sequences with a pre-trained model:

```bash
python -m lego_net.main --skip_training --load_model ./checkpoints/best_model.pt
```

## Example Outputs

After training, the model can generate various LEGO brick sequences from scratch:

- **Stack pattern**: Vertical towers
- **Row pattern**: Horizontal lines
- **Stair pattern**: Diagonal staircases
- **Random walk**: Irregular patterns with small variations

Generated structures are visualized in 3D and saved to the `./output` directory.

## Customization

- Modify `data_generator.py` to add new pattern types
- Adjust model parameters in `transformer.py` for different model sizes
- Experiment with different learning rates and training parameters

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper
- Thanks to the PyTorch team for the excellent deep learning framework 