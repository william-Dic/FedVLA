# FedVLA/DP/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import logging
from typing import Optional, Tuple

# Configure logging if run as main, otherwise assume it's configured elsewhere
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SinusoidalPosEmb(nn.Module):
    """
    Generates sinusoidal positional embeddings for the diffusion timestep.
    Taken from: https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    def __init__(self, dim: int):
        """
        Initializes the sinusoidal positional embedding module.

        Args:
            dim (int): The dimension of the embeddings to generate.
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Generates the embeddings.

        Args:
            time (torch.Tensor): A tensor of timesteps, shape (batch_size,).

        Returns:
            torch.Tensor: The generated embeddings, shape (batch_size, dim).
        """
        device = time.device
        half_dim = self.dim // 2
        # Handle potential division by zero if dim=0 or 1, although unlikely for embeddings
        if half_dim <= 1:
             denominator = 1.0 # Avoid log(10000) / 0
        else:
             denominator = half_dim - 1
        # Prevent potential overflow with large denominators
        if denominator == 0:
            embeddings = 0.0 # Or handle appropriately
        else:
            embeddings = math.log(10000) / denominator
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Ensure time is broadcastable: (batch_size,) -> (batch_size, 1)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1)) # Pad the last dimension
        return embeddings

class MLPBlock(nn.Module):
    """A simple MLP block with LayerNorm and GELU activation."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim), # LayerNorm before activation
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class DiffusionPolicyModel(nn.Module):
    """
    A diffusion policy model that predicts noise based on state, timestep,
    and image features extracted via ResNet-34.
    """
    def __init__(self,
                 state_dim: int,
                 time_emb_dim: int = 64,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 image_feature_dim: int = 512, # ResNet-34 output feature dim
                 use_pretrained_resnet: bool = True,
                 freeze_resnet: bool = True
                ):
        """
        Initializes the Diffusion Policy Model with ResNet-34 image backbone.

        Args:
            state_dim (int): The dimensionality of the input state vector
                             (e.g., 7 for 6 angles + 1 gripper).
            time_emb_dim (int, optional): Dimensionality of timestep embedding. Defaults to 64.
            hidden_dim (int, optional): Dimensionality of hidden layers. Defaults to 256.
            num_layers (int, optional): Number of MLP blocks. Defaults to 4.
            image_feature_dim (int, optional): Expected dimensionality of ResNet features.
                                               Defaults to 512 (ResNet-18/34).
            use_pretrained_resnet (bool, optional): Whether to load pretrained weights for ResNet.
                                                    Defaults to True.
            freeze_resnet (bool, optional): Whether to freeze ResNet weights during training.
                                            Defaults to True.
        """
        super().__init__()
        self.state_dim = state_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim
        self.image_feature_dim = image_feature_dim

        # --- Image Backbone (ResNet-34) ---
        weights = models.ResNet34_Weights.DEFAULT if use_pretrained_resnet else None
        resnet = models.resnet34(weights=weights)

        # Remove the final classification layer (fc) and the avg pooling layer
        # We'll add our own adaptive pooling
        self.image_backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Add adaptive average pooling to get a fixed-size output regardless of input image size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # --- CORRECTED WAY TO GET OUTPUT CHANNELS ---
        # Access the last block of the last layer (layer4)
        # For ResNet-18/34, the blocks are BasicBlock
        last_block = resnet.layer4[-1]
        # The output channels are determined by the second conv layer in the BasicBlock
        _resnet_output_channels = last_block.conv2.out_channels
        # --- END CORRECTION ---

        if _resnet_output_channels != image_feature_dim:
             logging.warning(f"Provided image_feature_dim ({image_feature_dim}) doesn't match "
                             f"ResNet-34 output channels ({_resnet_output_channels}). Using {_resnet_output_channels}.")
             self.image_feature_dim = _resnet_output_channels # Correct the dimension

        if freeze_resnet:
            for param in self.image_backbone.parameters():
                param.requires_grad = False
            logging.info("ResNet backbone weights frozen.")
        else:
             logging.info("ResNet backbone weights will be fine-tuned.")
        # --- End Image Backbone ---


        # --- Timestep Embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        # --- End Timestep Embedding ---


        # --- Main Policy Network ---
        # Input projection layer now takes state + image features
        input_proj_dim = state_dim + self.image_feature_dim
        self.input_projection = MLPBlock(input_proj_dim, hidden_dim)

        # MLP layers conditioned on time
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # Each layer takes the hidden state + time embedding
            self.layers.append(MLPBlock(hidden_dim + time_emb_dim, hidden_dim))

        # Final output layer to predict noise
        self.output_projection = nn.Linear(hidden_dim, state_dim)
        # --- End Main Policy Network ---

        logging.info(f"Initialized DiffusionPolicyModel:")
        logging.info(f"  State Dim: {state_dim}")
        logging.info(f"  Image Feature Dim: {self.image_feature_dim}")
        logging.info(f"  Time Emb Dim: {time_emb_dim}")
        logging.info(f"  Hidden Dim: {hidden_dim}")
        logging.info(f"  Num Layers: {num_layers}")
        logging.info(f"  Using Pretrained ResNet: {use_pretrained_resnet}")
        logging.info(f"  Freezing ResNet: {freeze_resnet}")


    def forward(self,
                state: torch.Tensor,
                timestep: torch.Tensor,
                image_input: torch.Tensor
               ) -> torch.Tensor:
        """
        Forward pass of the diffusion model.

        Args:
            state (torch.Tensor): The current state tensor, shape (batch_size, state_dim).
            timestep (torch.Tensor): The current diffusion timestep, shape (batch_size,).
            image_input (torch.Tensor): Batch of input images, expected shape
                                        (batch_size, 3, H, W). Should be normalized
                                        as expected by ResNet.

        Returns:
            torch.Tensor: The predicted noise, shape (batch_size, state_dim).
        """
        # Ensure inputs are on the same device (implicitly handled if model is on one device)
        # device = state.device

        # 1. Extract Image Features
        # image_input shape: (batch_size, 3, H, W)
        image_features = self.image_backbone(image_input) # (batch_size, C, H', W')
        image_features = self.adaptive_pool(image_features) # (batch_size, C, 1, 1)
        image_features = torch.flatten(image_features, 1) # (batch_size, C)

        # Ensure feature dim matches expected *after* flattening
        if image_features.shape[-1] != self.image_feature_dim:
             # This check might be redundant now due to the correction in __init__,
             # but kept for safety.
             raise ValueError(f"Internal error: ResNet output dim {image_features.shape[-1]} "
                              f"doesn't match expected {self.image_feature_dim}")

        # 2. Embed timestep
        time_embedding = self.time_mlp(timestep) # (batch_size, time_emb_dim)

        # 3. Concatenate state and image features
        # state shape: (batch_size, state_dim)
        combined_input = torch.cat([state, image_features], dim=-1) # (batch_size, state_dim + image_feature_dim)

        # 4. Project combined input to hidden dimension
        x = self.input_projection(combined_input) # (batch_size, hidden_dim)

        # 5. Process through layers, conditioning on time embedding
        for layer in self.layers:
            # Concatenate hidden state and time embedding
            input_to_layer = torch.cat([x, time_embedding], dim=-1) # (batch_size, hidden_dim + time_emb_dim)
            # Apply layer
            x = layer(input_to_layer) # (batch_size, hidden_dim)
            # Optional: Add residual connection: x = x + layer(input_to_layer)

        # 6. Project to output dimension (predict noise)
        predicted_noise = self.output_projection(x) # (batch_size, state_dim)

        return predicted_noise

# --- Example Usage ---
if __name__ == "__main__":
    # Configuration
    STATE_DIM = 7  # 6 joint angles + 1 gripper value
    IMAGE_H, IMAGE_W = 224, 224 # Example ResNet standard input size
    BATCH_SIZE = 4
    IMAGE_FEATURE_DIM = 512 # ResNet-34 output feature dim before FC layer

    # Instantiate the model
    model = DiffusionPolicyModel(
        state_dim=STATE_DIM,
        image_feature_dim=IMAGE_FEATURE_DIM,
        use_pretrained_resnet=True,
        freeze_resnet=True # Typically freeze backbone initially
    )

    print("-" * 30)
    # Note: Printing the full model is very verbose due to ResNet
    # print(f"Model Architecture:\n{model}")
    print("DiffusionPolicyModel with ResNet-34 backbone initialized.")
    print("-" * 30)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {num_total_params:,}")
    # Check if freezing worked by checking a parameter in the backbone
    is_frozen = not model.image_backbone[0].weight.requires_grad # Check conv1 weight
    print(f"Trainable Parameters: {num_params:,} (ResNet frozen: {is_frozen})")
    print("-" * 30)


    # Create dummy input data (ensure image tensor has correct shape and type)
    dummy_state = torch.randn(BATCH_SIZE, STATE_DIM)
    dummy_timestep = torch.randint(0, 1000, (BATCH_SIZE,)) # Example diffusion timesteps
    # Image input needs shape (B, C, H, W) and be float
    dummy_image_input = torch.randn(BATCH_SIZE, 3, IMAGE_H, IMAGE_W)

    # Perform a forward pass
    try:
        # Pass state, timestep, and image input to the model
        predicted_noise = model(dummy_state, dummy_timestep, dummy_image_input)

        print("Forward pass successful!")
        print(f"Input state shape: {dummy_state.shape}")
        print(f"Input timestep shape: {dummy_timestep.shape}")
        print(f"Input image shape: {dummy_image_input.shape}")
        print(f"Output predicted noise shape: {predicted_noise.shape}")

        # Check output dimension matches state dimension
        assert predicted_noise.shape == (BATCH_SIZE, STATE_DIM)
        print("Output shape matches state dimension. Basic check passed.")

    except Exception as e:
        print(f"Error during forward pass: {e}")
        logging.exception("Forward pass failed")

    print("-" * 30)
