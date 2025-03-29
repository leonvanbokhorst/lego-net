"""
Step-based prediction model for LEGO brick placement.
Instead of predicting absolute coordinates, this model predicts relative step vectors
between bricks, which works better for random walk patterns.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from lego_net.model.transformer import PositionalEncoding, TransformerDecoderLayer

class StepPredictorTransformer(nn.Module):
    """
    Transformer model that predicts step vectors instead of absolute coordinates.
    
    This model:
    1. Takes a sequence of coordinates as input
    2. Converts them to step vectors (differences between consecutive points)
    3. Predicts the next step vector
    4. Can convert back to absolute coordinates for generation
    """
    def __init__(
        self, 
        d_model: int = 64, 
        nhead: int = 8, 
        num_layers: int = 3, 
        dim_feedforward: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize the step-based LEGO transformer model.
        
        Args:
            d_model: Dimension of the model's embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer decoder layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
        """
        super().__init__()
        
        # Linear projection from 3D coordinates to d_model dimensions
        self.input_proj = nn.Linear(3, d_model)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection to predict step vector
        self.output_proj = nn.Linear(d_model, 3)
        
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a causal mask to prevent attention to future positions."""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def _convert_coords_to_steps(self, coords: torch.Tensor) -> torch.Tensor:
        """Convert absolute coordinates to step vectors."""
        # coords: [batch_size, seq_len, 3]
        steps = coords[:, 1:, :] - coords[:, :-1, :]
        return steps
    
    def _convert_steps_to_coords(self, steps: torch.Tensor, start_coord: torch.Tensor) -> torch.Tensor:
        """Convert step vectors back to absolute coordinates."""
        # steps: [batch_size, seq_len, 3]
        # start_coord: [batch_size, 1, 3]
        batch_size, seq_len, _ = steps.shape
        
        # Initialize with start_coord
        coords = torch.zeros(batch_size, seq_len + 1, 3, device=steps.device)
        coords[:, 0, :] = start_coord.squeeze(1)
        
        # Cumulatively add steps
        for i in range(seq_len):
            coords[:, i+1, :] = coords[:, i, :] + steps[:, i, :]
        
        return coords
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the transformer model.
        
        Args:
            x: Input sequence of brick coordinates [batch_size, seq_len, 3]
            
        Returns:
            Tuple of (predicted step vectors, predicted coordinates)
        """
        # x: [batch_size, seq_len, 3]
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Convert to steps
        if seq_len > 1:
            steps = self._convert_coords_to_steps(x)
            
            # Project to embedding dimensions
            emb = self.input_proj(steps)  # [batch_size, seq_len-1, d_model]
            
            # Add positional encoding
            emb = self.pos_enc(emb)
            
            # Generate attention mask
            mask = self._generate_square_subsequent_mask(seq_len-1).to(x.device)
            
            # Apply transformer decoder layers
            for layer in self.layers:
                emb = layer(emb, mask)
            
            # Final layer norm
            emb = self.norm(emb)
            
            # Project back to 3D step vectors
            pred_steps = self.output_proj(emb)  # [batch_size, seq_len-1, 3]
            
            # Convert predicted steps back to coordinates
            # Use the original coordinates except the first one as the base
            pred_coords = self._convert_steps_to_coords(
                pred_steps, 
                x[:, 1:2, :]  # Use second point as start point
            )
            
            # Shift pred_coords to align with original sequence
            pred_coords = torch.cat([
                x[:, :1, :],  # Keep the first original point
                pred_coords[:, :-1, :]  # Remove the last predicted point
            ], dim=1)
            
            return pred_steps, pred_coords
        else:
            # Not enough points to predict steps
            return torch.zeros(batch_size, 0, 3, device=x.device), x

    def generate_sequence(
        self, 
        start_seq: torch.Tensor, 
        num_steps: int
    ) -> torch.Tensor:
        """
        Generate a sequence by predicting steps and converting back to coordinates.
        
        Args:
            start_seq: Starting sequence of coordinates [batch_size, seq_len, 3]
            num_steps: Number of steps to generate
            
        Returns:
            Generated sequence [batch_size, seq_len + num_steps, 3]
        """
        self.eval()
        batch_size, seq_len, _ = start_seq.shape
        
        # Need at least 2 points to compute a step
        if seq_len < 2:
            raise ValueError("Need at least 2 points in starting sequence to generate")
        
        # Start with the input sequence
        current_seq = start_seq.clone()
        
        with torch.no_grad():
            for _ in range(num_steps):
                # Predict next step
                _, pred_coords = self.forward(current_seq)
                
                # Get the last predicted point
                next_point = pred_coords[:, -1:, :]
                
                # Append to sequence
                current_seq = torch.cat([current_seq, next_point], dim=1)
        
        return current_seq

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 