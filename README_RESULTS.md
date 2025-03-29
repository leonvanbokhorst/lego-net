# LEGO Brick Placement Transformer Models

## Project Summary
This project implements and compares three different neural network architectures for predicting LEGO brick placements in 3D space:

1. **Original Transformer** - A basic transformer decoder model that predicts the absolute coordinates of the next brick
2. **Random Walk Specific Model** - A specialized version of the original model trained specifically on random walk patterns
3. **Step-Based Predictor** - An advanced model that predicts relative step vectors between bricks rather than absolute positions

## Key Findings

### Model Performance Comparison
The step-based predictor significantly outperforms the other models across all pattern types:

| Pattern Type | Original Model | Random Walk Model | Step Predictor |
|--------------|----------------|-------------------|----------------|
| Random Walk  | 0.196552       | 0.128793          | **0.114754**   |
| Stack        | 0.898832       | 0.417523          | **0.001951**   | 
| Row          | 0.917026       | 3.756550          | **0.013034**   |
| Stair        | 2.268482       | 6.004579          | **0.007892**   |

Values represent Mean Squared Error (lower is better).

### Why Step-Based Prediction Works Better

1. **More Natural Representation**: Brick placement is inherently a relative process. Builders think in terms of "place the next brick 1 unit above the previous one" rather than "place a brick at absolute coordinates (3,4,5)".

2. **Better Random Walk Modeling**: Random walks are defined by their step distributions, not by absolute positions. By modeling the steps directly, we better capture the underlying pattern.

3. **Improved Pattern Recognition**: The step-based model can more easily identify common patterns like "always move up 1 unit" (stack) or "always move right 1 unit" (row) because these patterns are explicit in the step representation.

4. **Transferability**: A step-based model trained on one starting position can easily generalize to sequences starting at different locations.

## Visualizations

The `output/model_comparison` directory contains visualizations comparing how each model predicts different LEGO brick patterns. Key observations:

- The original model tends to converge to a standard pattern regardless of input
- The random walk specific model improves on random walks but performs worse on structured patterns
- The step-based model accurately captures the underlying pattern in all test cases

## Technical Implementation

- All models implemented using PyTorch
- The step-based predictor uses a novel approach of predicting step vectors rather than absolute coordinates
- The model architecture includes:
  - Transformer decoder with self-attention
  - Custom embedding layer
  - Specialized conversion between absolute coordinates and relative step vectors

## Conclusion

The step-based transformer model represents a significant improvement for LEGO brick sequence prediction. By focusing on the relations between bricks rather than their absolute positions, the model better captures the underlying patterns in LEGO constructions.

This approach could be extended to more complex LEGO models with varying brick types and orientations, potentially enabling automated LEGO construction planning and interactive building assistance. 