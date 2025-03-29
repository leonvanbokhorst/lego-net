# Minimal Transformer for LEGO Brick Placement

## Overview  
This project demonstrates a **small transformer-style neural network** that learns to generate sequences of 2×4 LEGO brick placements. We use a basic **Transformer decoder** architecture in PyTorch to predict the next brick’s position given previous bricks, treating brick positions as a sequence. The network is lightweight (small embedding size and few layers) to suit rapid prototyping and educational clarity. We train it on **synthetic data** of simple LEGO build patterns (stacking, rows, staircases, etc.), using a regression loss (MSE) since outputs are continuous coordinates. After training, the model can generate new brick sequences from scratch in an autoregressive manner. Below we outline data representation, synthetic data generation, the model architecture, training procedure, and how to visualize the generated sequences.

## Data Representation  
Each LEGO brick placement is represented as a token containing its **3D coordinates**. For simplicity, we use a tuple `(x, y, z)` (floating-point values) for the brick’s position. All bricks are identical 2×4 units, so orientation can be omitted or added later as an extra feature. A full LEGO build sequence is then a list of coordinate tokens in the order the bricks are placed. 

For model training, we will use sequences of fixed length (e.g. 6 bricks per sequence for illustration). The model will learn to predict the next brick’s coordinates at each step. This means if a training sequence is `[b1, b2, b3, ...]` (where each `b_i = (x_i, y_i, z_i)`), the network is trained such that given `[b1]` it predicts `b2`, given `[b1, b2]` it predicts `b3`, and so on. We prepare input-output pairs as follows: the **input sequence** is all bricks except the last, and the **target** is the same sequence shifted left by one (all bricks except the first). This way, at position *i* the model tries to output the coordinates of brick *i+1*. We use continuous coordinate values directly, so the network performs regression on each coordinate (using Mean Squared Error loss) rather than classification.

*Example:* A sequence of 5 bricks in a vertical stack (each new brick one unit higher in z) might be:  
```
[(0,0,0), (0,0,1), (0,0,2), (0,0,3), (0,0,4)]
```  
The input to the model would be `[(0,0,0), (0,0,1), (0,0,2), (0,0,3)]` and the target outputs would be `[(0,0,1), (0,0,2), (0,0,3), (0,0,4)]`. During generation, the model will be used autoregressively: we feed the first brick, get a prediction for the second, append it and feed again to get the third, and so on.

## Synthetic Data Generation  
We create a toy dataset of brick sequences using simple programmatically generated patterns. This synthetic data will allow the transformer to learn basic spatial patterns of brick placement. We consider a few pattern types to provide variety:

- **Vertical stack:** bricks placed directly on top of each other (constant x, y; increasing z).
- **Horizontal row:** bricks placed in a line (constant y, z; increasing x).
- **Staircase:** bricks placed in a rising diagonal (increasing both x and z, simulating a stair).
- **Random walk:** bricks placed with small random offsets in the plane and occasional upward moves (introducing some randomness).

Each pattern generates a sequence of coordinates. For example, a horizontal row might start at `(0,0,0)` and place each subsequent brick 1 unit to the right: `(1,0,0), (2,0,0)`, etc. A staircase might increment both x and z by 1 each step: `(0,0,0) -> (1,0,1) -> (2,0,2) -> ...`. Below is a code snippet to generate a sequence given a pattern name. We randomly choose a pattern for each training example to diversify the data:

```python
import torch
import numpy as np

def generate_brick_sequence(pattern=None, length=6):
    patterns = ["stack", "row", "stair", "random_walk"]
    if pattern is None:
        pattern = np.random.choice(patterns)
    x = y = z = 0.0  # start at origin
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

# Example usage:
print(generate_brick_sequence("stack", length=5))
# Output: tensor([[0.,0.,0.],[0.,0.,1.],[0.,0.,2.],[0.,0.,3.],[0.,0.,4.]])
```  

In practice we will generate a fresh batch of sequences on the fly each epoch (since the data is synthetic and essentially unlimited). This on-the-fly generation helps avoid overfitting to a small fixed set of sequences and keeps the training data varied.

## Transformer Model Architecture  
We implement a **minimal Transformer decoder** to process the sequence of brick coordinates. The core idea of a Transformer is to use **self-attention** to allow each position in the sequence to attend to (i.e., learn from) other positions. Because we want an autoregressive model (each prediction based only on past bricks), we apply a *causal mask* to the self-attention so that a position can only attend to earlier positions in the sequence (and not “cheat” by looking at future bricks). Our architecture consists of:

- **Input projection (embedding):** a linear layer that maps the 3-dimensional coordinate input into a higher-dimensional feature space (`d_model`), essentially the embedding vector for the brick’s position. Using a learned projection for continuous inputs lets the model transform raw coordinates into a space where attention can identify patterns.
- **Positional encoding:** added to the input embeddings to inform the model of each brick’s position index in the sequence. We use the standard sinusoidal positional encoding from the original Transformer paper ([Positional Encoding. This article is the second in The… | by Hunter Phillips | Medium](https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6#:~:text=To%20ensure%20each%20position%20has,are%20generated%20for%20each%20position)) – a fixed set of sine and cosine functions of different frequencies. This produces a unique vector for each position that we add to the embeddings. This way, the network can differentiate the *order* of bricks (e.g., first vs. third brick) and learn order-dependent patterns.
- **Transformer decoder layers:** one or more self-attention layers with multi-head attention. Each layer has:
  - A **Multi-Head Self-Attention** sublayer that allows the model to attend to all previous bricks in the sequence. We use PyTorch’s `nn.MultiheadAttention`. We supply an attention mask to ensure each position only attends to earlier positions (the mask is an upper-triangular matrix of `-inf` values that blocks future positions ([
Generating PyTorch Transformer Masks | James D. McCaffrey](https://jamesmccaffrey.wordpress.com/2020/12/16/generating-pytorch-transformer-masks/#:~:text=def%20generate_square_subsequent_mask%28sz%29%3A%20mask%20%3D%20%28T,0%29%29%20return%20mask))).
  - A **Feed-Forward Network (FFN)** sublayer (a small two-layer MLP) applied to each position’s output, to further transform the features non-linearly. We include ReLU activation in the FFN.
  - **Residual (skip) connections** around each sublayer and **layer normalization**, as per the Transformer design, to help training stability. This means the input to a sublayer is added to its output (so the layer only needs to learn a residual), followed by normalization.
- **Output projection:** a final linear layer that maps the transformer’s `d_model` features back down to 3 numbers, representing the predicted coordinates of the next brick. The output is interpreted as the model’s prediction for the *next* brick at each position of the input sequence.

We choose small dimensions for simplicity: e.g., `d_model = 32` (embedding size 32), `nhead = 4` (4 attention heads), and perhaps 1–2 decoder layers. The number of heads must divide `d_model` evenly (here each head would be size 8). A single decoder layer can already learn patterns; more layers can learn more complex dependencies at the cost of more computation. Below is the implementation of the model in PyTorch:

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        # Create positional encoding matrix of shape (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len,1]
        # Compute sinusoidal frequencies as per "Attention is All You Need"
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * 
                              (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # apply cosine to odd indices
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model] for broadcasting
        self.register_buffer('pe', pe)  # register as buffer so it gets correct device, not a parameter

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        # Add positional encoding to the input embeddings
        return x + self.pe[:, :seq_len, :]

class LegoTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64):
        super().__init__()
        # Linear projection from 3-dim coordinates to d_model dimensions
        self.input_proj = nn.Linear(3, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        # Define multiple decoder layers (self-attention + FFN)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, batch_first=True) for _ in range(num_layers)
        ])
        self.ff_layers1 = nn.ModuleList([nn.Linear(d_model, dim_feedforward) for _ in range(num_layers)])
        self.ff_layers2 = nn.ModuleList([nn.Linear(dim_feedforward, d_model) for _ in range(num_layers)])
        self.norm_layers1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm_layers2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        # Output linear layer to predict next coordinate (3 values)
        self.output_proj = nn.Linear(d_model, 3)

    def forward(self, x):
        # x: [batch_size, seq_len, 3] input sequence of coordinates
        x = self.input_proj(x)               # [batch_size, seq_len, d_model]
        x = self.pos_enc(x)                 # add positional encoding
        seq_len = x.size(1)
        # Causal mask to prevent attention to future positions
        attn_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        # Apply each Transformer decoder layer
        for attn, ff1, ff2, norm1, norm2 in zip(self.attn_layers, self.ff_layers1, 
                                                self.ff_layers2, self.norm_layers1, self.norm_layers2):
            # Self-attention: query, key, value are all x (since no encoder memory)
            attn_out, _ = attn(x, x, x, attn_mask=attn_mask)
            x = norm1(x + attn_out)         # add & norm (residual connection)
            # Feed-forward network
            ff_out = ff2(torch.relu(ff1(x)))
            x = norm2(x + ff_out)           # add & norm
        # Project to 3D coordinate output for each sequence position
        return self.output_proj(x)
```

A few implementation details: We used `batch_first=True` in `MultiheadAttention` so that inputs are shape `[batch, seq, features]` for convenience. The `attn_mask` is a square matrix with `-inf` (negative infinity) in positions where the target index is ahead of the source index, which effectively zeroes out attention to future tokens ([
Generating PyTorch Transformer Masks | James D. McCaffrey](https://jamesmccaffrey.wordpress.com/2020/12/16/generating-pytorch-transformer-masks/#:~:text=def%20generate_square_subsequent_mask%28sz%29%3A%20mask%20%3D%20%28T,0%29%29%20return%20mask)). The positional encoding uses the standard formula from Vaswani et al. (2017) ([Positional Encoding. This article is the second in The… | by Hunter Phillips | Medium](https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6#:~:text=To%20ensure%20each%20position%20has,are%20generated%20for%20each%20position)) and is registered as a buffer so it moves to GPU if the model is moved. We can adjust `num_layers`, `d_model`, etc., to make the model larger if needed, but the defaults here keep it minimal.

## Training Procedure  
We train the model using a simple supervised learning approach. For each training step, we generate a batch of synthetic sequences and train the model to predict the next brick positions. Key points in training:

- **Loss function:** We use Mean Squared Error (MSE) loss between the predicted coordinates and the actual next coordinates for each position in the sequence. MSE is appropriate for continuous regression outputs (as opposed to cross-entropy which is used for classification).
- **Optimizer:** We can use Adam for faster convergence on this toy problem. A modest learning rate (e.g. 0.001–0.01) works for our small network.
- **Batching:** We train on multiple sequences per batch to better estimate gradients. We generate, for example, 32 sequences of length 6 each time. We stack them into a tensor of shape `[batch, seq_len, 3]`. We then split this into `input` (all but last brick) and `target` (all but first brick) as described earlier, each of shape `[batch, seq_len-1, 3]`.
- **Epochs:** Since the data is synthetic, we can run for a number of epochs (each with new random sequences) until the loss stabilizes or decreases to a low value. Because the task is simple patterns, it might only take a few hundred iterations for the model to start mimicking the patterns.

Below is a training loop example. It generates random sequences each epoch, trains the model, and prints the loss periodically:

```python
# Initialize model, loss, optimizer
model = LegoTransformer(d_model=32, nhead=4, num_layers=2, dim_feedforward=64)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Training loop
epochs = 100  # for example
batch_size = 32
for epoch in range(1, epochs+1):
    # Generate a batch of random sequences
    sequences = [generate_brick_sequence(length=6) for _ in range(batch_size)]
    batch = torch.stack(sequences, dim=0)        # shape: [batch_size, 6, 3]
    inp = batch[:, :-1, :]   # input: first 5 bricks
    tgt = batch[:, 1:, :]    # target: next 5 bricks
    # Forward pass
    pred = model(inp)        # pred shape: [batch_size, 5, 3]
    loss = loss_fn(pred, tgt)
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Print loss occasionally
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Training MSE: {loss.item():.4f}")
```

Each epoch, the network sees new sequences which include different patterns. Over time, it should learn the underlying rules (e.g., bricks in a row have constant y,z and increasing x; in a stack, x,y constant and z increases, etc.). As training progresses, the loss (average squared prediction error in coordinates) should decrease. On this toy problem, a well-tuned small model can achieve a low MSE where predicted positions closely match the true next positions.

## Generating New Sequences  
After training, we can use the model to **generate a sequence of LEGO brick placements from scratch.** This is done by feeding the model its own prior outputs in a loop (autoregressive generation):

1. **Start with an initial token:** e.g., the first brick at `(0,0,0)` or any starting position you choose. This is our current sequence.
2. **Predict the next brick:** Feed the current sequence into the model. The model will output a prediction for the next brick at the last position of the output sequence.
3. **Append the predicted brick** to the sequence.
4. **Repeat**: use the extended sequence as new input to predict the subsequent brick.

We continue this until the sequence reaches a desired length or some stopping criterion. (In more advanced setups, one could also have a special end-of-sequence token, but for simplicity we generate a fixed number of bricks.)

Here’s how you could implement sequence generation with the trained model:

```python
model.eval()  # set to evaluation mode
start_brick = torch.tensor([[0.0, 0.0, 0.0]])            # a single brick (1,3) tensor
generated_sequence = [start_brick]                      # list to store generated bricks (tensor shape [1,3] each)
num_generate = 5                                        # how many more bricks to generate
for step in range(num_generate):
    # Prepare current sequence tensor of shape [1, current_length, 3]
    current_seq = torch.stack(generated_sequence, dim=1)  # note: list of [1,3] -> tensor [1, len, 3]
    # Get model prediction for the next brick
    with torch.no_grad():
        pred_seq = model(current_seq)       # output shape [1, current_length, 3]
    next_brick = pred_seq[:, -1, :]         # take the last predicted brick (shape [1,3])
    generated_sequence.append(next_brick.squeeze(0))   # append as [3] tensor for next loop

# Convert generated_sequence to a nice list of coordinates
generated_coords = [tuple(brick.tolist()) for brick in generated_sequence]
print("Generated brick sequence:", generated_coords)
```

For example, if the model learned to imitate a “row” pattern, a generated sequence might look like:  
`[(0.0,0.0,0.0), (0.98, 0.05, 0.02), (1.99, -0.03, -0.01), (3.01, 0.04, 0.03), ...]` which is roughly a line in x (with slight variations). Rounding those, it’s essentially `(0,0,0) -> (1,0,0) -> (2,0,0) -> (3,0,0) ...`. A staircase pattern generation might produce increasing x and z coordinates. Because our model outputs continuous values, the numbers may not be exact integers but should be close to the learned pattern.

## Visualization of LEGO Sequences  
To **visualize the generated sequences**, we can use simple plotting. One approach is to use Matplotlib’s 3D plotting capabilities to scatter plot the brick coordinates and even draw lines or annotated points to show the sequence order:

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

coords = np.array(generated_coords)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=range(len(coords)), cmap='viridis', s=50)
for i, (x,y,z) in enumerate(coords):
    ax.text(x, y, z, str(i), color='red')  # label each brick by index in sequence
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
plt.title("Generated LEGO brick sequence")
plt.show()
```

This will show the bricks in 3D space, with a color gradient or index labels indicating the order they were placed. For instance, a stack will show points aligned vertically; a row will show points along the X-axis, etc. Using a color map (`c=range(len(coords))`) can indicate progression (e.g., darker to lighter as index increases). 

For a more LEGO-specific visualization, one could integrate with a 3D rendering (like using `pyplot.voxels` to draw cubes, or exporting the coordinates to a LEGO CAD software format). However, the scatter plot is a quick way to confirm the model’s behavior.

## Conclusion  
We built a minimal Transformer decoder that learns to output the next brick placement given a sequence of previous bricks. Despite its simplicity, this prototype captures basic spatial patterns (like linear or vertical placements) from synthetic data. Key components included representing bricks as tokens with continuous coordinates, sinusoidal positional encoding to handle sequence order, and a masked multi-head attention mechanism to enable autoregressive predictions. We used MSE loss to train on coordinate outputs, and demonstrated how to generate new sequences and visualize them. This framework can be extended with orientation prediction, longer sequences, or more complex transformer layers for more sophisticated LEGO structure generation.