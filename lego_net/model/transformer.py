import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention is All You Need" paper.
    Adds position information to the input embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        
        # Compute sinusoidal frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * 
                            (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension for broadcasting
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings of shape [batch_size, seq_len, d_model]
            
        Returns:
            Embeddings with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LegoTransformer(nn.Module):
    """
    Transformer model for LEGO brick sequence prediction.
    
    Uses a minimal transformer decoder architecture with self-attention
    to predict the next position of a brick given a sequence of previous
    brick placements.
    """
    def __init__(
        self, 
        d_model: int = 32, 
        nhead: int = 4, 
        num_layers: int = 2, 
        dim_feedforward: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize the LEGO transformer model.
        
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
        
        # Output projection to predict next brick coordinates
        self.output_proj = nn.Linear(d_model, 3)
        
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a mask to prevent attention to future positions.
        
        Args:
            sz: Sequence length
            
        Returns:
            A square attention mask (sz, sz) where elements in the upper triangle
            are -inf and lower triangle are 0.
        """
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer model.
        
        Args:
            x: Input sequence of brick coordinates [batch_size, seq_len, 3]
            
        Returns:
            Predictions for the next brick at each position [batch_size, seq_len, 3]
        """
        # x: [batch_size, seq_len, 3]
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Project 3D coordinates to embedding dimensions
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_enc(x)
        
        # Generate attention mask
        mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Apply transformer decoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final layer norm
        x = self.norm(x)
        
        # Project back to 3D coordinates
        output = self.output_proj(x)  # [batch_size, seq_len, 3]
        
        return output


class TransformerDecoderLayer(nn.Module):
    """
    A single transformer decoder layer with self-attention and feed-forward network.
    """
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through a single transformer decoder layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor of the same shape
        """
        # Self-attention block
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward block
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model with a simple example
    model = LegoTransformer(d_model=32, nhead=4, num_layers=2)
    print(f"Model has {count_parameters(model):,} trainable parameters")
    
    # Create dummy input
    batch_size, seq_len = 2, 5
    x = torch.randn(batch_size, seq_len, 3)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}") 