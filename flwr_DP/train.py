# FedVLA/DP/train.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import logging
import argparse
import math
from typing import Optional, List, Tuple, Union, Dict

# Import custom modules
from dataset import RobotEpisodeDataset # Assuming dataset.py is in the same directory
from model import DiffusionPolicyModel  # Assuming model.py is in the same directory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Diffusion Schedule Helpers ---

def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Generates a linear schedule for beta values."""
    return torch.linspace(beta_start, beta_end, timesteps)

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Extracts the appropriate schedule values for a batch of timesteps t."""
    batch_size = t.shape[0]
    # Ensure t has the same device as a for gather
    out = a.to(t.device).gather(-1, t)
    # Reshape for broadcasting
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# --- Forward Diffusion Process (Adding Noise) ---

def q_sample(x_start: torch.Tensor, t: torch.Tensor, sqrt_alphas_cumprod: torch.Tensor, sqrt_one_minus_alphas_cumprod: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Adds noise to the data x_start according to the timestep t."""
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    noisy_x = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_x

# --- Reverse Diffusion Process (Sampling/Denoising) ---

@torch.no_grad() # Sampling doesn't require gradients
def p_sample(model: nn.Module, x: torch.Tensor, t: torch.Tensor, t_index: int,
             betas: torch.Tensor, sqrt_one_minus_alphas_cumprod: torch.Tensor,
             sqrt_recip_alphas: torch.Tensor, posterior_variance: torch.Tensor,
             image_input: torch.Tensor) -> torch.Tensor:
    """
    Performs one step of the DDPM reverse process (sampling).
    x_{t-1} ~ p(x_{t-1} | x_t)

    Args:
        model: The diffusion model.
        x (torch.Tensor): The noisy state at timestep t (x_t), shape (batch_size, state_dim).
        t (torch.Tensor): The current timestep t for the batch, shape (batch_size,).
        t_index (int): The integer index corresponding to timestep t.
        betas: Precomputed schedule tensor.
        sqrt_one_minus_alphas_cumprod: Precomputed schedule tensor.
        sqrt_recip_alphas: Precomputed schedule tensor.
        posterior_variance: Precomputed schedule tensor.
        image_input (torch.Tensor): The conditioning image input, shape (batch_size, C, H, W).

    Returns:
        torch.Tensor: The estimated state at timestep t-1 (x_{t-1}).
    """
    # Use model to predict noise added at step t
    predicted_noise = model(state=x, timestep=t, image_input=image_input)

    # Get schedule values for timestep t
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Calculate the mean of the posterior p(x_{t-1} | x_t, x_0)
    # This is the core DDPM sampling equation
    mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        # If t=0, the sample is deterministic (no noise added)
        return mean
    else:
        # If t > 0, add noise based on the posterior variance
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4: x_{t-1} = mean + sqrt(posterior_variance) * noise
        return mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model: nn.Module, shape: tuple, timesteps: int,
                  betas: torch.Tensor, sqrt_one_minus_alphas_cumprod: torch.Tensor,
                  sqrt_recip_alphas: torch.Tensor, posterior_variance: torch.Tensor,
                  device: torch.device, image_input: torch.Tensor) -> torch.Tensor:
    """
    Performs the full DDPM sampling loop, starting from noise.

    Args:
        model: The diffusion model.
        shape (tuple): The desired shape of the output tensor (batch_size, state_dim).
        timesteps (int): Total number of diffusion steps.
        betas, ...: Precomputed schedule tensors.
        device: The device to perform sampling on.
        image_input (torch.Tensor): The conditioning image input for the batch.

    Returns:
        torch.Tensor: The final denoised sample (predicted x_0).
    """
    batch_size = shape[0]

    # Start from pure noise (x_T)
    img = torch.randn(shape, device=device)

    # Iterate backwards from T-1 down to 0
    for i in tqdm(reversed(range(0, timesteps)), desc="Sampling", total=timesteps, leave=False):
        # Create timestep tensor for the current index i
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        # Perform one denoising step
        img = p_sample(model, img, t, i,
                       betas, sqrt_one_minus_alphas_cumprod,
                       sqrt_recip_alphas, posterior_variance,
                       image_input) # Pass conditioning image

    # img now holds the predicted x_0
    return img


# --- Custom Collate Function ---
# (Keep the existing custom_collate_fn as is)
def custom_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[None, None]]:
    """
    Custom collate function to handle batching of (state, image_tensor) tuples.
    Filters out invalid items and returns (None, None) if the resulting batch would be empty.
    """
    states = []
    images = []
    valid_item_count = 0
    for i, item in enumerate(batch):
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            # logging.warning(f"Skipping malformed batch item at index {i}: Type {type(item)}, Value {item}")
            continue
        state, image = item
        if not (isinstance(state, torch.Tensor) and isinstance(image, torch.Tensor)):
             # logging.warning(f"Skipping batch item at index {i} with non-tensor element: state type {type(state)}, image type {type(image)}")
             continue
        states.append(state)
        images.append(image)
        valid_item_count += 1

    if valid_item_count == 0:
        if batch:
             logging.warning(f"Collate function resulted in an empty batch after filtering {len(batch)} items. Check dataset __getitem__ for errors.")
        return None, None

    try:
        states_batch = torch.stack(states, dim=0)
        images_batch = torch.stack(images, dim=0)
    except Exception as e:
         logging.error(f"Error stacking {valid_item_count} valid tensors in collate_fn: {e}. ")
         for i in range(min(3, len(states))):
             logging.error(f"  Sample {i}: state shape {states[i].shape}, image shape {images[i].shape}")
         return None, None

    return states_batch, images_batch

# --- Evaluation Function (Modified for Sampling) ---

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device,
             diffusion_timesteps: int, betas: torch.Tensor,
             sqrt_one_minus_alphas_cumprod: torch.Tensor, sqrt_recip_alphas: torch.Tensor,
             posterior_variance: torch.Tensor, num_eval_samples: int) -> float:
    """
    Evaluates the model by sampling predictions and comparing to ground truth.

    Args:
        model: The diffusion policy model.
        dataloader: DataLoader for the evaluation dataset.
        device: The device to run evaluation on (CPU or GPU).
        diffusion_timesteps, betas, ...: Schedule tensors needed for sampling.
        num_eval_samples (int): The number of samples to generate and evaluate.

    Returns:
        Average Mean Squared Error (MSE) between predicted and ground truth states.
    """
    model.eval() # Set model to evaluation mode
    total_mse = 0.0
    samples_evaluated = 0

    with torch.no_grad(): # Disable gradient calculations
        progress_bar = tqdm(dataloader, desc="Evaluating (Sampling)", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            if samples_evaluated >= num_eval_samples:
                break # Stop after evaluating the desired number of samples

            if batch is None or batch == (None, None):
                logging.warning(f"Skipping empty/invalid batch during evaluation at index {batch_idx}.")
                continue

            try:
                gt_state_batch, image_batch = batch # Ground Truth state
            except Exception as e:
                 logging.error(f"Error unpacking evaluation batch {batch_idx}: {e}")
                 continue

            if gt_state_batch is None or image_batch is None:
                 logging.warning(f"Skipping evaluation batch {batch_idx} due to None tensor.")
                 continue

            # Determine how many samples from this batch to evaluate
            batch_size = gt_state_batch.shape[0]
            samples_to_take = min(batch_size, num_eval_samples - samples_evaluated)
            if samples_to_take <= 0: continue # Should not happen with outer check, but safety

            # Select subset of the batch if needed
            gt_state_batch = gt_state_batch[:samples_to_take].to(device)
            image_batch = image_batch[:samples_to_take].to(device)

            if gt_state_batch.shape[0] == 0 or image_batch.shape[0] == 0:
                 logging.warning(f"Skipping empty evaluation batch {batch_idx} after device transfer/subsetting.")
                 continue

            # --- Perform Sampling ---
            logging.info(f"Generating {gt_state_batch.shape[0]} samples for evaluation batch {batch_idx}...")
            predicted_state_batch = p_sample_loop(
                model,
                shape=gt_state_batch.shape, # Shape of the state tensor
                timesteps=diffusion_timesteps,
                betas=betas,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                sqrt_recip_alphas=sqrt_recip_alphas,
                posterior_variance=posterior_variance,
                device=device,
                image_input=image_batch # Pass the conditioning image
            )
            logging.info(f"Sample generation finished for batch {batch_idx}.")


            # --- Calculate Metric (MSE) ---
            mse = F.mse_loss(predicted_state_batch, gt_state_batch, reduction='sum') # Sum MSE over batch
            total_mse += mse.item()
            samples_evaluated += gt_state_batch.shape[0]

            # Log comparison for the first sample in the batch
            if batch_idx == 0 and samples_evaluated > 0:
                 logging.info("--- Evaluation Sample Comparison (First Batch) ---")
                 logging.info(f"Ground Truth State (first sample): {gt_state_batch[0].cpu().numpy()}")
                 logging.info(f"Predicted State (first sample):  {predicted_state_batch[0].cpu().numpy()}")
                 logging.info("-------------------------------------------------")

            progress_bar.set_postfix(samples=f"{samples_evaluated}/{num_eval_samples}")


    model.train() # Set model back to training mode

    if samples_evaluated == 0:
        logging.warning("Evaluation sampling completed without evaluating any samples.")
        return float('inf') # Or handle as appropriate

    avg_mse = total_mse / samples_evaluated # Calculate average MSE per sample
    return avg_mse


# --- Training Function ---

def train(args):
    """
    Main training loop for the diffusion policy model.
    """
    # --- Device Setup ---
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Diffusion Schedule Setup ---
    timesteps = args.diffusion_timesteps
    betas = linear_beta_schedule(timesteps=timesteps, beta_start=args.beta_start, beta_end=args.beta_end).to(device) # Move schedule to device
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    # Precompute values needed for q_sample AND p_sample
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    # Calculate posterior variance q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    # Clip variance to avoid issues at t=0 (though p_sample handles t=0 separately)
    # posterior_variance_clipped = torch.clamp(posterior_variance, min=1e-20)

    logging.info(f"Diffusion schedule set up with {timesteps} timesteps.")

    # --- Transforms ---
    image_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Training Dataset and DataLoader ---
    try:
        import inspect
        sig = inspect.signature(RobotEpisodeDataset.__init__)
        dataset_args = {'base_dir': args.data_dir, 'num_episodes': args.num_episodes}
        if 'transform' in sig.parameters: dataset_args['transform'] = image_transform
        train_dataset = RobotEpisodeDataset(**dataset_args)
    except Exception as e:
        logging.exception(f"Error initializing training dataset from {args.data_dir}")
        return
    if len(train_dataset) == 0: logging.error("Training dataset is empty."); return
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False,
                                  collate_fn=custom_collate_fn)
    logging.info(f"Training dataset loaded: {len(train_dataset)} samples.")

    # --- Evaluation Dataset and DataLoader ---
    eval_dataloader = None
    if args.eval_interval > 0: # Only load if eval_interval is set
        eval_data_dir = args.eval_data_dir if args.eval_data_dir else args.data_dir
        eval_num_episodes = args.eval_num_episodes if args.eval_num_episodes else args.num_episodes
        if eval_data_dir == args.data_dir and eval_num_episodes > args.num_episodes:
            eval_num_episodes = args.num_episodes
        try:
            eval_dataset_args = {'base_dir': eval_data_dir, 'num_episodes': eval_num_episodes}
            if 'transform' in sig.parameters: eval_dataset_args['transform'] = image_transform
            eval_dataset = RobotEpisodeDataset(**eval_dataset_args)
        except Exception as e:
            logging.exception(f"Error initializing evaluation dataset from {eval_data_dir}. Disabling evaluation.")
            eval_dataset = None

        if eval_dataset and len(eval_dataset) > 0:
            eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False,
                                         collate_fn=custom_collate_fn)
            logging.info(f"Evaluation dataset loaded: {len(eval_dataset)} samples.")
        else:
            if eval_dataset is None: logging.warning("Evaluation dataset failed to load. Skipping evaluation.")
            else: logging.warning("Evaluation dataset is empty. Skipping evaluation.")
            eval_dataloader = None
    else:
        logging.info("Evaluation interval is 0. Skipping evaluation setup.")


    # --- Model Initialization ---
    model = DiffusionPolicyModel(
        state_dim=args.state_dim, time_emb_dim=args.time_emb_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_mlp_layers, image_feature_dim=args.image_feature_dim,
        use_pretrained_resnet=args.use_pretrained_resnet, freeze_resnet=args.freeze_resnet
    ).to(device)
    logging.info("Model initialized.")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable parameters: {num_params:,}")

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    logging.info(f"Optimizer: AdamW (lr={args.learning_rate}, weight_decay={args.weight_decay})")

    # --- Loss Function (for training) ---
    train_criterion = nn.MSELoss()
    logging.info("Training loss function: MSELoss (on noise)")

    # --- Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Model checkpoints will be saved to: {args.output_dir}")

    # --- Training Loop ---
    logging.info("Starting training...")
    global_step = 0
    best_eval_metric = float('inf') # Now tracking MSE

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        batches_processed_this_epoch = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=True)

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            if batch is None or batch == (None, None): continue
            try: state_batch, image_batch = batch
            except Exception as e: logging.error(f"Error unpacking training batch {batch_idx}: {e}"); continue
            if state_batch is None or image_batch is None: continue

            try:
                state_batch = state_batch.to(device)
                image_batch = image_batch.to(device)
            except Exception as e: logging.error(f"Error moving training batch {batch_idx} to device {device}: {e}"); continue
            if state_batch.shape[0] == 0 or image_batch.shape[0] == 0: continue

            # --- Training Step (predict noise) ---
            current_batch_size = state_batch.shape[0]
            t = torch.randint(0, timesteps, (current_batch_size,), device=device).long()
            noise = torch.randn_like(state_batch)
            noisy_state_batch = q_sample(
                x_start=state_batch, t=t,
                sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                noise=noise
            )
            predicted_noise = model(
                state=noisy_state_batch, timestep=t, image_input=image_batch
            )
            loss = train_criterion(predicted_noise, noise) # Use training criterion (MSE on noise)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            batches_processed_this_epoch += 1
            progress_bar.set_postfix(loss=loss.item())


        # --- End of Epoch ---
        if batches_processed_this_epoch > 0:
             avg_epoch_loss = epoch_loss / batches_processed_this_epoch
             logging.info(f"Epoch {epoch+1}/{args.num_epochs} Training Avg Loss (Noise MSE): {avg_epoch_loss:.4f}")
        else:
             logging.warning(f"Epoch {epoch+1}/{args.num_epochs} completed without processing any training batches.")
             avg_epoch_loss = float('inf')

        # --- Evaluation Step (Sampling) ---
        if eval_dataloader is not None and (epoch + 1) % args.eval_interval == 0:
            logging.info(f"--- Starting evaluation sampling for Epoch {epoch+1} ({args.num_eval_samples} samples) ---")
            avg_eval_mse = evaluate( # Now returns MSE
                model, eval_dataloader, device,
                timesteps, betas, sqrt_one_minus_alphas_cumprod,
                sqrt_recip_alphas, posterior_variance, args.num_eval_samples
            )
            logging.info(f"Epoch {epoch+1}/{args.num_epochs} Evaluation Avg State MSE: {avg_eval_mse:.4f}")

            # Save best model based on eval MSE (lower is better)
            if avg_eval_mse < best_eval_metric:
                best_eval_metric = avg_eval_mse
                best_checkpoint_path = os.path.join(args.output_dir, "model_best.pth")
                try:
                    torch.save({
                        'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'eval_metric': avg_eval_mse, # Store eval MSE
                        'args': vars(args)
                    }, best_checkpoint_path)
                    logging.info(f"Saved new best model checkpoint to {best_checkpoint_path} (Eval State MSE: {best_eval_metric:.4f})")
                except Exception as e:
                     logging.error(f"Failed to save best checkpoint at epoch {epoch+1}: {e}")
            logging.info(f"--- Finished evaluation sampling for Epoch {epoch+1} ---")


        # --- Save Regular Checkpoint ---
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.num_epochs:
            checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
            try:
                torch.save({
                    'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_epoch_loss, # Store training loss
                    'args': vars(args)
                }, checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                 logging.error(f"Failed to save checkpoint at epoch {epoch+1}: {e}")


    logging.info("Training finished.")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diffusion Policy Model")

    # Paths and Directories
    parser.add_argument('--data_dir', type=str, default='../stack_orange/', help='Base directory for training dataset')
    parser.add_argument('--eval_data_dir', type=str, default=None, help='Base directory for evaluation dataset (uses data_dir if None)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--num_episodes', type=int, default=95, help='Number of episodes to load for training')
    parser.add_argument('--eval_num_episodes', type=int, default=None, help='Number of episodes for evaluation (uses num_episodes if None)')

    # Model Hyperparameters
    parser.add_argument('--state_dim', type=int, default=7, help='Dimension of the state vector')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to')
    parser.add_argument('--image_feature_dim', type=int, default=512, help='Feature dimension from ResNet backbone')
    parser.add_argument('--time_emb_dim', type=int, default=64, help='Dimension for timestep embedding')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for MLP layers')
    parser.add_argument('--num_mlp_layers', type=int, default=4, help='Number of MLP layers')
    parser.add_argument('--use_pretrained_resnet', action='store_true', help='Use pretrained ResNet weights')
    parser.add_argument('--no_use_pretrained_resnet', action='store_false', dest='use_pretrained_resnet', help='Do not use pretrained ResNet weights')
    parser.add_argument('--freeze_resnet', action='store_true', help='Freeze ResNet backbone weights')
    parser.add_argument('--no_freeze_resnet', action='store_false', dest='freeze_resnet', help='Do not freeze ResNet backbone weights')
    parser.set_defaults(use_pretrained_resnet=True, freeze_resnet=True)

    # Diffusion Hyperparameters
    parser.add_argument('--diffusion_timesteps', type=int, default=1000, help='Total number of diffusion timesteps')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Starting value for linear beta schedule')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Ending value for linear beta schedule')

    # Training Hyperparameters
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Batch size for evaluation sampling (often needs to be smaller due to sampling loop memory)') # Reduced eval batch size
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Optimizer weight decay')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader')
    parser.add_argument('--save_interval', type=int, default=50, help='Save checkpoint every N epochs')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluate model every N epochs (set to 0 to disable)')
    parser.add_argument('--num_eval_samples', type=int, default=64, help='Number of samples to generate during evaluation') # Added num_eval_samples
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use if available')


    args = parser.parse_args()

    if os.name == 'nt' and args.num_workers > 0:
         logging.warning("Setting num_workers > 0 on Windows can cause issues. Forcing num_workers = 0.")
         args.num_workers = 0

    # Ensure eval_interval > 0 if we want evaluation
    if args.eval_interval <= 0:
         logging.info("Evaluation interval is <= 0. Evaluation will be disabled.")
         args.eval_dataloader = None # Explicitly disable

    logging.info("Training arguments:")
    for arg, value in sorted(vars(args).items()):
        logging.info(f"  {arg}: {value}")

    train(args)
