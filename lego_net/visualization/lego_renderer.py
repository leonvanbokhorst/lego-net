#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LEGO brick renderer for turning coordinates into nice LEGO brick visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
import os

def create_brick(position, color='red', alpha=0.8, brick_size=(1, 2, 0.5)):
    """
    Create a 3D representation of a LEGO brick at the given position.
    
    Args:
        position: (x, y, z) position of the brick's center
        color: Color of the brick
        alpha: Transparency of the brick
        brick_size: Size of the brick (x, y, z) in LEGO units
            Default is (1, 2, 0.5) which is a 1x2 flat brick
        
    Returns:
        vertices and faces for rendering
    """
    # Get position
    x, y, z = position
    
    # Define the size of a brick
    dx, dy, dz = brick_size
    
    # Create the 8 vertices of the brick
    vertices = np.array([
        [x - dx/2, y - dy/2, z],           # bottom face
        [x + dx/2, y - dy/2, z],
        [x + dx/2, y + dy/2, z],
        [x - dx/2, y + dy/2, z],
        [x - dx/2, y - dy/2, z + dz],      # top face
        [x + dx/2, y - dy/2, z + dz],
        [x + dx/2, y + dy/2, z + dz],
        [x - dx/2, y + dy/2, z + dz]
    ])
    
    # Define the 6 faces using the vertices
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
    ]
    
    # Add studs on top (simplified approach to avoid index errors)
    # Just add a simple cylinder for each stud instead of the complex approach
    stud_radius = 0.2
    stud_height = 0.15
    
    # Create a single stud in the center of the brick
    stud_x = x
    stud_y = y
    stud_z = z + dz  # Top of the brick
    
    # Create a simple box for the stud instead of a detailed cylinder
    stud_vertices = [
        [stud_x - stud_radius, stud_y - stud_radius, stud_z],                    # bottom square
        [stud_x + stud_radius, stud_y - stud_radius, stud_z],
        [stud_x + stud_radius, stud_y + stud_radius, stud_z],
        [stud_x - stud_radius, stud_y + stud_radius, stud_z],
        [stud_x - stud_radius, stud_y - stud_radius, stud_z + stud_height],      # top square
        [stud_x + stud_radius, stud_y - stud_radius, stud_z + stud_height],
        [stud_x + stud_radius, stud_y + stud_radius, stud_z + stud_height],
        [stud_x - stud_radius, stud_y + stud_radius, stud_z + stud_height]
    ]
    
    # Create the 6 faces of the stud box
    stud_faces = [
        [stud_vertices[0], stud_vertices[1], stud_vertices[2], stud_vertices[3]],  # bottom
        [stud_vertices[4], stud_vertices[5], stud_vertices[6], stud_vertices[7]],  # top
        [stud_vertices[0], stud_vertices[1], stud_vertices[5], stud_vertices[4]],  # front
        [stud_vertices[2], stud_vertices[3], stud_vertices[7], stud_vertices[6]],  # back
        [stud_vertices[0], stud_vertices[3], stud_vertices[7], stud_vertices[4]],  # left
        [stud_vertices[1], stud_vertices[2], stud_vertices[6], stud_vertices[5]]   # right
    ]
    
    return faces + stud_faces, color, alpha


def render_lego_sequence(coords, output_path=None, title="LEGO Brick Sequence", brick_colors=None):
    """
    Render a sequence of LEGO bricks from coordinates.
    
    Args:
        coords: Array of shape (n, 3) containing the (x, y, z) coordinates of n bricks
        output_path: Path to save the visualization
        title: Title for the plot
        brick_colors: List of colors for the bricks, if None, a sequence of colors will be used
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use colormap if colors not provided
    if brick_colors is None:
        cmap = plt.cm.viridis
        colors = [cmap(i / len(coords)) for i in range(len(coords))]
    else:
        colors = brick_colors
    
    # Default LEGO colors if needed
    default_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
    
    # Get coordinate ranges for proper scaling
    x_min, y_min, z_min = np.min(coords, axis=0)
    x_max, y_max, z_max = np.max(coords, axis=0)
    
    # Create bricks
    for i, (position, color) in enumerate(zip(coords, colors)):
        # Use a color from the default list if needed
        if isinstance(color, (float, int)):
            color = default_colors[i % len(default_colors)]
            
        # Create the brick
        faces, face_color, alpha = create_brick(position, color, 0.8)
        
        # Add to plot
        collection = Poly3DCollection(faces, alpha=alpha)
        collection.set_facecolor(face_color)
        collection.set_edgecolor('black')
        ax.add_collection3d(collection)
        
        # Add label with the brick number
        ax.text(position[0], position[1], position[2] + 0.7, 
                str(i), color='black', fontsize=12, ha='center')
    
    # Set axis limits with some padding
    padding = 2  # Add padding around the bricks
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_zlim(z_min - padding, z_max + padding)
    
    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    
    # Set equal aspect ratio
    # This is important for LEGO bricks to look proportional
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    mid_x = (x_max + x_min) / 2
    mid_y = (y_max + y_min) / 2
    mid_z = (z_max + z_min) / 2
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Save if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
    return fig, ax


def visualize_model_comparison(model_outputs, pattern_name, output_dir):
    """
    Create a comparison visualization of different model outputs as LEGO bricks.
    
    Args:
        model_outputs: Dictionary of model_name -> coordinates array
        pattern_name: Name of the pattern being visualized
        output_dir: Directory to save the visualization
    """
    # Create a figure with subplots for each model
    num_models = len(model_outputs)
    fig = plt.figure(figsize=(15, 5 * ((num_models + 1) // 2)))
    
    # Color scheme for different patterns
    color_schemes = {
        'stack': ['red', 'darkred', 'indianred', 'firebrick', 'lightcoral'],
        'row': ['blue', 'navy', 'royalblue', 'steelblue', 'skyblue'],
        'stair': ['green', 'darkgreen', 'seagreen', 'mediumseagreen', 'lightgreen'],
        'random_walk': ['purple', 'indigo', 'darkviolet', 'mediumorchid', 'plum']
    }
    
    default_colors = color_schemes.get(pattern_name, 
                                      ['orange', 'darkorange', 'peru', 'goldenrod', 'gold'])
    
    # Plot each model's output
    for i, (model_name, coords) in enumerate(model_outputs.items()):
        ax = fig.add_subplot(num_models, 1, i+1, projection='3d')
        
        # Generate a color sequence
        colors = []
        for j in range(len(coords)):
            idx = min(j, len(default_colors)-1)
            colors.append(default_colors[idx])
        
        # Create bricks
        for j, (position, color) in enumerate(zip(coords, colors)):
            # Create the brick
            faces, face_color, alpha = create_brick(position, color, 0.8)
            
            # Add to plot
            collection = Poly3DCollection(faces, alpha=alpha)
            collection.set_facecolor(face_color)
            collection.set_edgecolor('black')
            ax.add_collection3d(collection)
            
            # Add label with the brick number
            ax.text(position[0], position[1], position[2] + 0.7, 
                    str(j), color='black', fontsize=10, ha='center')
        
        # Set axis limits based on data range
        x_min, y_min, z_min = np.min(coords, axis=0)
        x_max, y_max, z_max = np.max(coords, axis=0)
        padding = 2
        
        # Set equal aspect ratio
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{model_name.capitalize()} Model - {pattern_name.capitalize()} Pattern")
    
    plt.tight_layout()
    
    # Save the comparison
    output_path = os.path.join(output_dir, f"lego_{pattern_name}_comparison.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Test with some example data
    coords = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],
        [1, 0, 2],
        [1, 0, 3]
    ])
    
    # Create a test visualization
    fig, ax = render_lego_sequence(coords, "test_lego_render.png", "Test LEGO Sequence")
    plt.show() 