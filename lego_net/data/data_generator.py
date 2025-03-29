import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Union, Literal

def generate_brick_sequence(
    pattern: Optional[Literal["stack", "row", "stair", "random_walk"]] = None, 
    length: int = 6,
    start_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> torch.Tensor:
    """
    Generate a synthetic sequence of LEGO brick placements.
    
    Args:
        pattern: The type of pattern to generate. If None, a random pattern is chosen.
        length: The number of bricks in the sequence.
        start_pos: The starting position (x, y, z) for the first brick.
        
    Returns:
        A tensor of shape [length, 3] with the coordinates of each brick.
    """
    patterns = ["stack", "row", "stair", "random_walk"]
    if pattern is None:
        pattern = np.random.choice(patterns)
        
    x, y, z = start_pos
    seq = []
    
    if pattern == "stack":
        # Constant x,y; increment z each step
        for i in range(length):
            seq.append([x, y, z])
            z += 1.0  # move up by 1 unit
    elif pattern == "row":
        # Constant y,z; increment x each step
        for i in range(length):
            seq.append([x, y, z])
            x += 1.0  # move in x direction
    elif pattern == "stair":
        # Increment both x and z to form a staircase
        for i in range(length):
            seq.append([x, y, z])
            x += 1.0  # move right
            z += 1.0  # move up
    elif pattern == "random_walk":
        # Small random steps in x,y and occasional step in z
        for i in range(length):
            seq.append([x, y, z])
            x += np.random.uniform(-0.5, 0.5)
            y += np.random.uniform(-0.5, 0.5)
            z += 1.0 if np.random.rand() > 0.7 else 0.0
            
    return torch.tensor(seq, dtype=torch.float32)

def generate_batch(
    batch_size: int = 32, 
    seq_length: int = 6, 
    pattern: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of sequences and prepare input-output pairs for training.
    
    Args:
        batch_size: Number of sequences in the batch.
        seq_length: Length of each sequence.
        pattern: If specified, all sequences will follow this pattern.
               Otherwise, random patterns are chosen for each sequence.
               
    Returns:
        input_batch: Tensor of shape [batch_size, seq_length-1, 3] for inputs
        target_batch: Tensor of shape [batch_size, seq_length-1, 3] for targets
    """
    # Generate sequences
    sequences = [generate_brick_sequence(pattern, seq_length) for _ in range(batch_size)]
    batch = torch.stack(sequences, dim=0)  # shape: [batch_size, seq_length, 3]
    
    # Split into input and target pairs
    input_batch = batch[:, :-1, :]   # All except last brick
    target_batch = batch[:, 1:, :]   # All except first brick
    
    return input_batch, target_batch

def visualize_sequence(
    coords: Union[List[Tuple[float, float, float]], torch.Tensor, np.ndarray],
    show_indices: bool = True,
    title: str = "LEGO Brick Sequence",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a sequence of LEGO brick placements in 3D.
    
    Args:
        coords: List or array of (x, y, z) coordinates.
        show_indices: Whether to show indices next to each brick.
        title: Plot title.
        save_path: If provided, save the figure to this path.
    """
    # Convert input to numpy array if it's not already
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    elif isinstance(coords, list):
        coords = np.array(coords)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with color gradient based on sequence order
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=range(len(coords)), cmap='viridis', 
        s=100, alpha=0.8
    )
    
    # Add index labels if requested
    if show_indices:
        for i, (x, y, z) in enumerate(coords):
            ax.text(x, y, z, f" {i}", fontsize=12, color='red')
    
    # Add lines connecting bricks in sequence
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'k--', alpha=0.4)
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Z', fontsize=14)
    plt.title(title, fontsize=16)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Sequence Order', fontsize=12)
    
    # Set equal aspect ratio
    max_range = max([
        np.max(coords[:, 0]) - np.min(coords[:, 0]),
        np.max(coords[:, 1]) - np.min(coords[:, 1]),
        np.max(coords[:, 2]) - np.min(coords[:, 2])
    ])
    mid_x = (np.max(coords[:, 0]) + np.min(coords[:, 0])) / 2
    mid_y = (np.max(coords[:, 1]) + np.min(coords[:, 1])) / 2
    mid_z = (np.max(coords[:, 2]) + np.min(coords[:, 2])) / 2
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

# Test the module if run directly
if __name__ == "__main__":
    # Generate and visualize various patterns
    for pattern in ["stack", "row", "stair", "random_walk"]:
        seq = generate_brick_sequence(pattern, length=8)
        visualize_sequence(seq, title=f"Pattern: {pattern}") 