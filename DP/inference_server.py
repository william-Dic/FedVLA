import os
import socket
import pickle
import struct
import base64
import io
import logging
import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Attempt to import model and sampling functions
try:
    from model import DiffusionPolicyModel
    from train import linear_beta_schedule, p_sample_loop
except ImportError as e:
    logging.error(f"Error importing model or sampling functions: {e}")
    exit(1)


def send_msg(sock: socket.socket, msg: object) -> None:
    """
    Send a Python object via pickle with a 4-byte length prefix.
    """
    data = pickle.dumps(msg)
    length = struct.pack('>I', len(data))
    sock.sendall(length + data)


def recvall(sock: socket.socket, n: int) -> bytes:
    """
    Helper to recv n bytes or return None if EOF.
    """
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_msg(sock: socket.socket) -> object:
    """
    Receive a pickled object with a 4-byte length prefix.
    """
    raw_len = recvall(sock, 4)
    if raw_len is None:
        return None
    msg_len = struct.unpack('>I', raw_len)[0]
    data = recvall(sock, msg_len)
    return pickle.loads(data)


def main(args):
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load checkpoint
    if not os.path.isfile(args.checkpoint_path):
        logging.error(f"Checkpoint not found: {args.checkpoint_path}")
        return
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    train_args = checkpoint.get('args', vars(args))
    if isinstance(train_args, argparse.Namespace):
        train_args = vars(train_args)
    logging.info("Checkpoint loaded and arguments recovered.")

    # Diffusion schedule setup
    timesteps = train_args.get('diffusion_timesteps', args.diffusion_timesteps)
    betas = linear_beta_schedule(
        timesteps=timesteps,
        beta_start=train_args.get('beta_start', args.beta_start),
        beta_end=train_args.get('beta_end', args.beta_end)
    ).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    logging.info(f"Diffusion schedule with {timesteps} timesteps prepared.")

    # Model initialization
    state_dim = train_args['state_dim']
    model = DiffusionPolicyModel(
        state_dim=state_dim,
        time_emb_dim=train_args['time_emb_dim'],
        hidden_dim=train_args['hidden_dim'],
        num_layers=train_args['num_mlp_layers'],
        image_feature_dim=train_args['image_feature_dim'],
        use_pretrained_resnet=train_args.get('use_pretrained_resnet', True),
        freeze_resnet=train_args.get('freeze_resnet', True)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logging.info("Model loaded and set to eval mode.")

    # Image preprocessing
    image_size = train_args.get('image_size', args.image_size)
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Visualization setup
    if args.visualize:
        plt.ion()

    # Socket server setup
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((args.host, args.port))
    server.listen(1)
    logging.info(f"Server listening on {args.host}:{args.port}")

    conn, addr = server.accept()
    logging.info(f"Connection established from {addr}")

    try:
        while True:
            # Receive data from client
            msg = recv_msg(conn)
            if msg is None:
                logging.info("Client disconnected.")
                break

            # Parse input dict
            angles = msg.get('angles', [])
            gripper = msg.get('gripper_value', [0.0])
            img_b64 = msg.get('image', '')

            # Prepare state tensor
            state_list = angles + [float(gripper[0])]
            state_tensor = torch.tensor(state_list, dtype=torch.float32).unsqueeze(0).to(device)

            # Decode and preprocess image
            img_data = base64.b64decode(img_b64)
            pil_img = Image.open(io.BytesIO(img_data)).convert('RGB')
            img_tensor = image_transform(pil_img).unsqueeze(0).to(device)

            # Optional visualization
            if args.visualize:
                plt.imshow(pil_img)
                plt.title("Input Image")
                plt.axis('off')
                plt.pause(args.vis_pause_duration)

            # Diffusion inference
            with torch.no_grad():
                pred_batch = p_sample_loop(
                    model,
                    shape=(1, state_dim),
                    timesteps=timesteps,
                    betas=betas,
                    sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                    sqrt_recip_alphas=sqrt_recip_alphas,
                    posterior_variance=posterior_variance,
                    device=device,
                    image_input=img_tensor
                )

            # Extract prediction and send back
            pred_np = pred_batch.squeeze(0).cpu().numpy().tolist()
            send_msg(conn, {'action': pred_np})

    except KeyboardInterrupt:
        logging.info("Interrupted by user, shutting down.")
    finally:
        conn.close()
        server.close()
        logging.info("Server closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference Service for Diffusion Policy Model")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host/IP to bind the server')
    parser.add_argument('--port', type=int, default=50007, help='Port to bind the server')
    parser.add_argument('--checkpoint_path', type=str, default="../checkpoints/model_best.pth", help='Path to model checkpoint (.pth)')
    parser.add_argument('--visualize', action='store_true', help='Show input images during inference')
    parser.add_argument('--vis_pause_duration', type=float, default=0.05, help='Pause duration between frames when visualizing')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize input images')
    parser.add_argument('--diffusion_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Beta start value')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta end value')
    args = parser.parse_args()
    main(args)

