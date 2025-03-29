import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional, Union, Callable
import time
import os
from tqdm import tqdm

from lego_net.data.data_generator import generate_batch
from lego_net.model.transformer import LegoTransformer


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    batch_size: int,
    seq_length: int,
    num_batches: int,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        optimizer: The optimizer to use
        loss_fn: The loss function
        batch_size: Batch size
        seq_length: Sequence length
        num_batches: Number of batches per epoch
        device: Device to use (CPU/GPU)
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    epoch_loss = 0.0
    
    # Training loop
    for _ in tqdm(range(num_batches), desc="Training batches"):
        # Generate a batch of random sequences
        inputs, targets = generate_batch(batch_size, seq_length)
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / num_batches


def evaluate(
    model: nn.Module,
    loss_fn: Callable,
    batch_size: int,
    seq_length: int,
    num_batches: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on random test data.
    
    Args:
        model: The model to evaluate
        loss_fn: The loss function
        batch_size: Batch size
        seq_length: Sequence length
        num_batches: Number of batches to evaluate on
        device: Device to use (CPU/GPU)
        
    Returns:
        Tuple of (average loss, average MSE per coordinate)
    """
    model.eval()
    total_loss = 0.0
    total_mse_per_coord = 0.0
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Evaluation batches"):
            # Generate a batch of random sequences
            inputs, targets = generate_batch(batch_size, seq_length)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Calculate MSE per coordinate
            mse_per_coord = torch.mean((outputs - targets) ** 2).item()
            
            total_loss += loss.item()
            total_mse_per_coord += mse_per_coord
    
    return total_loss / num_batches, total_mse_per_coord / num_batches


def generate_sequence(
    model: nn.Module,
    start_brick: torch.Tensor,
    num_generate: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a sequence of bricks using the model.
    
    Args:
        model: The trained model
        start_brick: Starting brick coordinates (shape [1, 1, 3])
        num_generate: Number of bricks to generate
        device: Device to use (CPU/GPU)
        
    Returns:
        Tensor of shape [1, num_generate+1, 3] with the generated sequence
    """
    model.eval()
    current_seq = start_brick.to(device)
    
    with torch.no_grad():
        for _ in range(num_generate):
            # Predict next brick
            pred_seq = model(current_seq)
            next_brick = pred_seq[:, -1:, :]  # Shape [1, 1, 3]
            
            # Append prediction to sequence
            current_seq = torch.cat([current_seq, next_brick], dim=1)
    
    return current_seq


def train_model(
    model: nn.Module,
    epochs: int = 100,
    batch_size: int = 32,
    seq_length: int = 6,
    batches_per_epoch: int = 50,
    eval_batches: int = 10,
    learning_rate: float = 0.001,
    save_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """
    Train the LEGO transformer model.
    
    Args:
        model: The model to train
        epochs: Number of epochs
        batch_size: Batch size
        seq_length: Length of each sequence
        batches_per_epoch: Number of batches per epoch
        eval_batches: Number of batches to use for evaluation
        learning_rate: Learning rate
        save_dir: Directory to save checkpoints (None = don't save)
        device: Device to use (None = auto-detect)
        
    Returns:
        Dictionary with training history
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    # Create save directory if needed
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse_per_coord': []
    }
    
    # Training loop
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Train for one epoch
        train_loss = train_epoch(
            model, optimizer, loss_fn, batch_size, seq_length, 
            batches_per_epoch, device
        )
        
        # Evaluate
        val_loss, val_mse = evaluate(
            model, loss_fn, batch_size, seq_length, 
            eval_batches, device
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mse_per_coord'].append(val_mse)
        
        # Print progress
        print(f"Epoch {epoch}/{epochs}: "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Val MSE/coord: {val_mse:.6f}")
        
        # Save checkpoint if this is the best model so far
        if save_dir and val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
    
    # Save final model
    if save_dir:
        final_path = os.path.join(save_dir, 'final_model.pt')
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history
        }, final_path)
        print(f"Saved final model to {final_path}")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    return history


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot the training history.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot, or None to display
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MSE per coordinate
    plt.subplot(1, 2, 2)
    plt.plot(history['val_mse_per_coord'])
    plt.title('Validation MSE per Coordinate')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    
    plt.show()


def load_model(
    path: str, 
    model: Optional[nn.Module] = None, 
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a saved model checkpoint.
    
    Args:
        path: Path to the checkpoint file
        model: Model instance to load the state into, or None to create a new one
        device: Device to load the model onto
        
    Returns:
        Tuple of (loaded model, checkpoint data)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    
    # Create model if not provided
    if model is None:
        model = LegoTransformer()  # Create with default parameters
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, checkpoint 