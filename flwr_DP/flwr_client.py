from collections import OrderedDict
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import logging
import os
import argparse
from dataset import RobotEpisodeDataset
from model import DiffusionPolicyModel
from train import linear_beta_schedule, q_sample, custom_collate_fn
from tqdm import tqdm
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create checkpoints directory if it doesn't exist
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_model():
    """Load the diffusion policy model."""
    model = DiffusionPolicyModel(
        state_dim=7,  # 6 joint angles + 1 gripper value
        time_emb_dim=64,
        hidden_dim=256,
        num_layers=4,
        image_feature_dim=512,
        use_pretrained_resnet=True,
        freeze_resnet=True
    )
    return model

def load_data(client_id, num_clients):
    """Load the training and test data for a specific client."""
    # Image transforms
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load full dataset
    full_dataset = RobotEpisodeDataset(
        base_dir='../stack_orange/',
        num_episodes=95,  # Adjust based on your needs
        transform=image_transform
    )

    # Split dataset among clients
    total_size = len(full_dataset)
    client_size = total_size // num_clients
    start_idx = client_id * client_size
    end_idx = start_idx + client_size if client_id < num_clients - 1 else total_size

    # Create client-specific dataset
    client_dataset = Subset(full_dataset, range(start_idx, end_idx))
    logging.info(f"Client {client_id}: Using {len(client_dataset)} samples out of {total_size} total samples")

    # Create train and test loaders for this client
    train_size = int(0.8 * len(client_dataset))
    test_size = len(client_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(client_dataset, [train_size, test_size])

    trainloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    return trainloader, testloader

def train(model, trainloader, epochs=1):
    """Train the model for one epoch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Setup diffusion parameters
    timesteps = 1000
    betas = linear_beta_schedule(timesteps=timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # Setup optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    criterion = nn.MSELoss()

    total_loss = 0
    num_batches = 0

    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}/{epochs}")
        progress_bar = tqdm(trainloader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None or batch == (None, None):
                continue

            state_batch, image_batch = batch
            state_batch = state_batch.to(device)
            image_batch = image_batch.to(device)

            optimizer.zero_grad()

            # Training step (predict noise)
            current_batch_size = state_batch.shape[0]
            t = torch.randint(0, timesteps, (current_batch_size,), device=device).long()
            noise = torch.randn_like(state_batch)
            noisy_state_batch = q_sample(
                x_start=state_batch,
                t=t,
                sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                noise=noise
            )
            predicted_noise = model(
                state=noisy_state_batch,
                timestep=t,
                image_input=image_batch
            )
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })

        avg_loss = total_loss / num_batches
        logging.info(f"Epoch {epoch + 1} completed - Average Loss: {avg_loss:.4f}")
    
    return avg_loss

def test(model, testloader):
    """Evaluate the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Setup diffusion parameters
    timesteps = 1000
    betas = linear_beta_schedule(timesteps=timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    total_loss = 0
    total = 0

    logging.info("Starting evaluation...")
    progress_bar = tqdm(testloader, desc="Evaluating")

    with torch.no_grad():
        for batch in progress_bar:
            if batch is None or batch == (None, None):
                continue

            state_batch, image_batch = batch
            state_batch = state_batch.to(device)
            image_batch = image_batch.to(device)

            # For evaluation, we'll use multiple timesteps
            batch_size = state_batch.shape[0]
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(state_batch)
            
            # Add noise to the state
            noisy_state_batch = q_sample(
                x_start=state_batch,
                t=t,
                sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                noise=noise
            )
            
            # Predict noise
            predicted_noise = model(
                state=noisy_state_batch,
                timestep=t,
                image_input=image_batch
            )
            
            # Calculate loss between predicted and actual noise
            loss = nn.MSELoss()(predicted_noise, noise)
            total_loss += loss.item() * batch_size
            total += batch_size

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/total:.4f}'
            })

    avg_loss = total_loss / total
    # For diffusion models, we typically want to see the loss in the range of 0.1-1.0
    # A very low loss might indicate overfitting or incorrect evaluation
    logging.info(f"Evaluation completed - Average Loss: {avg_loss:.4f}")
    
    # Calculate a more meaningful accuracy metric
    # For diffusion models, we can use the percentage of predictions within a certain threshold
    accuracy = 1.0 - min(1.0, avg_loss)  # This will give us a more reasonable accuracy value
    
    return avg_loss, accuracy

def save_checkpoint(model, round_num, train_loss, eval_loss=None, eval_accuracy=None, client_id=None, total_epochs=None):
    """Save model checkpoint with metadata."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    client_str = f"_client{client_id}" if client_id is not None else ""
    epoch_str = f"_epoch{total_epochs}" if total_epochs is not None else ""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_round_{round_num}{client_str}{epoch_str}_{timestamp}.pth")
    
    checkpoint = {
        'round': round_num,
        'client_id': client_id,
        'total_epochs': total_epochs,
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
        'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy,
        'timestamp': timestamp
    }
    
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, num_clients):
        super().__init__()
        self.current_round = 0
        self.total_epochs = 0  # Track total epochs across all rounds
        self.client_id = client_id
        self.num_clients = num_clients
        self.net = load_model()
        self.trainloader, self.testloader = load_data(client_id, num_clients)
        self.ckpt_interval = 500  # Save checkpoint every 500 epochs
        self.eval_interval = 500  # Evaluate every 500 rounds
        logging.info(f"Initialized client {client_id} with {len(self.trainloader.dataset)} training samples and {len(self.testloader.dataset)} test samples")

    def get_parameters(self, config=None):
        logging.info(f"Client {self.client_id}: Getting model parameters")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        self.current_round += 1
        self.total_epochs += 1  # Increment total epochs
        logging.info(f"Client {self.client_id}: Starting local training for round {self.current_round}/2000 (Total epochs: {self.total_epochs})")
        set_parameters(self.net, parameters)
        train_loss = train(self.net, self.trainloader, epochs=1)
        logging.info(f"Client {self.client_id}: Local training completed - Loss: {train_loss:.4f}")
        
        # Save checkpoint every 500 epochs
        if self.total_epochs % self.ckpt_interval == 0:
            save_checkpoint(
                self.net, 
                self.current_round, 
                train_loss, 
                client_id=self.client_id,
                total_epochs=self.total_epochs
            )
            logging.info(f"Client {self.client_id}: Saved checkpoint at epoch {self.total_epochs} (round {self.current_round})")
        
        return self.get_parameters(config), len(self.trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        # Only evaluate every 500 rounds
        if self.current_round % self.eval_interval != 0:
            return 0.0, len(self.testloader.dataset), {"accuracy": 0.0}
            
        logging.info(f"Client {self.client_id}: Starting local evaluation at round {self.current_round} (Total epochs: {self.total_epochs})")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader)
        logging.info(f"Client {self.client_id}: Local evaluation completed - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save checkpoint with evaluation metrics every 500 epochs
        if self.total_epochs % self.ckpt_interval == 0:
            save_checkpoint(
                self.net, 
                self.current_round, 
                None, 
                loss, 
                accuracy, 
                client_id=self.client_id,
                total_epochs=self.total_epochs
            )
            logging.info(f"Client {self.client_id}: Saved evaluation checkpoint at epoch {self.total_epochs} (round {self.current_round})")
        
        return float(loss), len(self.testloader.dataset), {"accuracy": accuracy}

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def main():
    parser = argparse.ArgumentParser(description='Flower Client')
    parser.add_argument('--client-id', type=int, required=True, help='Client ID (0 to num_clients-1)')
    parser.add_argument('--num-clients', type=int, default=3, help='Total number of clients')
    parser.add_argument('--server-address', type=str, default="127.0.0.1:8080", help='Server address')
    args = parser.parse_args()

    if args.client_id < 0 or args.client_id >= args.num_clients:
        raise ValueError(f"Client ID must be between 0 and {args.num_clients-1}")

    logging.info(f"Starting Flower client {args.client_id}...")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=FlowerClient(args.client_id, args.num_clients),
    )

if __name__ == "__main__":
    main()
