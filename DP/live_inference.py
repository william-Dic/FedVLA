# FedVLA/DP/live_inference.py (Modified for Live Camera, Robot Control, Fixed Port & Baud)

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
import logging
import argparse
import math
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Union, Dict
import matplotlib.pyplot as plt
import time

# --- Robot Control Imports ---
import serial
import serial.tools.list_ports # Keep for potential future use or listing info
from pymycobot.mycobot import MyCobot
# --- End Robot Control Imports ---

# Import custom modules
try:
    from model import DiffusionPolicyModel
    from train import linear_beta_schedule, extract, p_sample, p_sample_loop
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure model.py and train.py (or the necessary functions) are accessible.")
    print("You may also need 'opencv-python', 'pyserial', 'pymycobot':")
    print("pip install opencv-python pyserial pymycobot")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'

# --- Fixed Robot Parameters ---
ROBOT_PORT = "/dev/ttyAMA0"
ROBOT_BAUD = 1000000
# --- End Fixed Robot Parameters ---

# --- Robot Setup Function ---
def setup_robot(port: str, baud: int = ROBOT_BAUD) -> Optional[MyCobot]:
    """Initializes the MyCobot connection on the specified port and baud rate."""
    global mc # Make mc global or pass it around
    mc = None
    logging.info(f"Attempting to connect to MyCobot on fixed port: {port} at fixed baud: {baud}.")

    try:
        mc = MyCobot(port, baudrate=baud)
        # Perform a quick check to see if the connection is live
        angles = mc.get_angles()
        if angles is None or not isinstance(angles, list):
             # Depending on the system, AMA0 might take a moment. Add a small delay and retry.
             time.sleep(0.5)
             angles = mc.get_angles()
             if angles is None or not isinstance(angles, list):
                 logging.warning(f"Connected to {port}, but failed to get initial angles after delay. Check robot power/connection/permissions.")
                 # return None # Option to exit if initial check fails
             else:
                 logging.info("Successfully retrieved angles after short delay.")

        logging.info(f"Successfully connected to MyCobot on {port}.")
        logging.info(f"Initial robot angles: {angles}") # Log initial state
        return mc
    except serial.SerialException as e:
        logging.error(f"Failed to connect to MyCobot on port {port}: {e}. Check permissions (e.g., user in 'dialout' or 'tty' group) and wiring.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during robot setup on port {port}: {e}")
        return None
# --- End Robot Setup Function ---


def run_live_inference(args, mc: MyCobot):
    """
    Loads a trained model, runs inference using a live camera feed,
    and sends predicted actions to the connected MyCobot.
    """
    # --- Device Setup ---
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Checkpoint ---
    if not os.path.isfile(args.checkpoint_path):
        logging.error(f"Checkpoint file not found: {args.checkpoint_path}")
        return
    logging.info(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Load training arguments
    train_args = checkpoint.get('args')
    if train_args is None:
        logging.warning("Checkpoint missing training arguments ('args'). Using CLI args as fallback.")
        train_args = vars(args)
    elif isinstance(train_args, argparse.Namespace):
        train_args = vars(train_args)
    logging.info("Loaded/fallback training arguments determined.")

    # --- Diffusion Schedule Setup ---
    try:
        # Use .get() to provide default values if missing from checkpoint/args fallback
        timesteps = train_args.get('diffusion_timesteps', args.diffusion_timesteps)
        beta_start = train_args.get('beta_start', args.beta_start)
        beta_end = train_args.get('beta_end', args.beta_end)
    except KeyError as e: # Should be less likely now with .get()
        logging.error(f"Missing critical diffusion parameter '{e}' even after fallback.")
        return
    betas = linear_beta_schedule(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    logging.info(f"Diffusion schedule set up with {timesteps} timesteps.")

    # --- Model Initialization ---
    try:
        # Use .get() for robustness against missing args in checkpoint
        state_dim = train_args.get('state_dim', args.state_dim)
        if state_dim != 7: # Specific check for this setup
            logging.warning(f"State_dim is {state_dim}, expected 7 (6 angles + 1 gripper). Ensure model matches robot.")

        time_emb_dim = train_args.get('time_emb_dim', args.time_emb_dim)
        hidden_dim = train_args.get('hidden_dim', args.hidden_dim)
        num_mlp_layers = train_args.get('num_mlp_layers', args.num_mlp_layers)
        image_feature_dim = train_args.get('image_feature_dim', args.image_feature_dim)
        use_pretrained_resnet = train_args.get('use_pretrained_resnet', args.use_pretrained_resnet)
        freeze_resnet = train_args.get('freeze_resnet', args.freeze_resnet)
    except KeyError as e: # Should be less likely now with .get()
        logging.error(f"Missing critical model parameter '{e}' even after fallback.")
        return

    model = DiffusionPolicyModel(
        state_dim=state_dim,
        time_emb_dim=time_emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_mlp_layers,
        image_feature_dim=image_feature_dim,
        use_pretrained_resnet=use_pretrained_resnet,
        freeze_resnet=freeze_resnet
    ).to(device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("Successfully loaded model state dict.")
    except Exception as e:
        logging.exception(f"Error loading state dict: {e}")
        return
    model.eval()

    # --- Image Transform ---
    image_size = train_args.get('image_size', args.image_size)
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    logging.info(f"Image transform set up for size {image_size}x{image_size}.")

    # --- Camera Setup ---
    logging.info(f"Attempting to open camera ID: {args.camera_id}")
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        logging.error(f"Error: Could not open camera with ID {args.camera_id}.")
        # Attempt to release if partially opened
        if cap: cap.release()
        return
    logging.info("Camera opened successfully.")
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Optional: Set camera resolution
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --- Visualization Setup ---
    vis_fig, vis_ax = None, None
    if args.visualize_trajectory:
        logging.info("Visualization enabled. Press 'q' in the window to quit.")
        plt.ion()
        vis_fig, vis_ax = plt.subplots(1, 1, figsize=(6, 6))
        vis_fig.suptitle("Live Camera Feed & Prediction")

    # --- Inference Loop (Live + Robot Control) ---
    frame_count = 0
    robot_speed = args.robot_speed # Use speed from arguments
    logging.info(f"Using robot command speed: {robot_speed}")

    try:
        with torch.no_grad():
            while True:
                loop_start_time = time.time()

                # --- Capture Frame ---
                ret, frame = cap.read()
                if not ret:
                    logging.error("Error: Can't receive frame (stream end?). Exiting ...")
                    break

                # --- Process Image ---
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                try:
                    image_tensor = image_transform(pil_image).unsqueeze(0).to(device)
                except Exception as e:
                    logging.warning(f"Skipping frame {frame_count} due to image transform error: {e}.")
                    continue

                # --- Visualize (Optional) ---
                if args.visualize_trajectory and vis_ax is not None and vis_fig is not None:
                    try:
                        vis_ax.clear()
                        vis_ax.imshow(pil_image)
                        vis_ax.set_title(f"Live View (Frame: {frame_count}) - Press 'q' to quit")
                        vis_ax.axis('off')
                        plt.draw()
                        plt.pause(0.001) # Minimal pause for GUI update
                        if not plt.fignum_exists(vis_fig.number):
                           logging.info("Visualization window closed. Exiting.")
                           break
                    except Exception as e:
                        logging.warning(f"Error during visualization update: {e}")
                        # Option to disable visualization if it keeps failing
                        # args.visualize_trajectory = False
                        # plt.ioff()
                        # plt.close(vis_fig)
                        # vis_fig, vis_ax = None, None


                # --- Perform Sampling ---
                inference_start_time = time.time()
                predicted_state_batch = p_sample_loop(
                    model, shape=(1, state_dim), timesteps=timesteps, betas=betas,
                    sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                    sqrt_recip_alphas=sqrt_recip_alphas, posterior_variance=posterior_variance,
                    device=device, image_input=image_tensor
                )
                inference_time = time.time() - inference_start_time

                # --- Process Prediction ---
                pred_state_np = predicted_state_batch.squeeze(0).cpu().numpy()

                # Extract angles and gripper value (assuming state_dim=7)
                if len(pred_state_np) == state_dim:
                    predicted_angles = pred_state_np[:-1].tolist() # List of 6 angles
                    predicted_gripper = pred_state_np[-1]          # Single gripper value

                    # Clamp gripper value to a reasonable range (e.g., 0-100)
                    predicted_gripper_clamped = np.clip(predicted_gripper, 0, 100)
                    # Optional: Clamp angles if needed, e.g., np.clip(angle, -170, 170) for each

                    print(f"\n--- Frame {frame_count} ---")
                    np.set_printoptions(precision=3, suppress=True)
                    # print(f"  Raw Predicted State: {pred_state_np}") # Can be verbose
                    print(f"  Predicted Angles:   {predicted_angles}")
                    print(f"  Predicted Gripper:  {predicted_gripper:.2f} (Clamped: {predicted_gripper_clamped:.0f})")
                    print(f"  Inference Time:     {inference_time:.4f} seconds")

                    # --- !!! SEND COMMANDS TO ROBOT !!! ---
                    try:
                        # Ensure mc object is valid before sending commands
                        if mc and hasattr(mc, 'send_angles') and hasattr(mc, 'set_gripper_value'):
                            mc.send_angles(predicted_angles, robot_speed)
                            # Ensure gripper value is an integer for the library
                            mc.set_gripper_value(int(round(predicted_gripper_clamped)), robot_speed)
                            # logging.debug(f"Sent state to robot: Angles={predicted_angles}, Gripper={int(round(predicted_gripper_clamped))}")
                        else:
                            logging.error("Robot object (mc) is invalid. Cannot send commands.")
                            # Optional: break or implement retry logic here
                            time.sleep(1) # Avoid spamming errors
                    except Exception as e:
                        logging.error(f"Error sending commands to robot: {e}")
                        # Consider pausing or stopping if robot communication fails repeatedly
                        time.sleep(0.5) # Small delay after error
                    # --- !!! END ROBOT CONTROL !!! ---

                else:
                    logging.warning(f"Predicted state dimension ({len(pred_state_np)}) does not match expected ({state_dim}). Skipping robot command.")


                frame_count += 1
                loop_time = time.time() - loop_start_time
                print(f"  Total Loop Time:    {loop_time:.4f} seconds")

                # Optional: Enforce a minimum loop time if needed
                # time.sleep(max(0, args.min_loop_duration - loop_time))


    except KeyboardInterrupt:
        logging.info("Keyboard interrupt detected. Exiting.")
    finally:
        # --- Cleanup ---
        logging.info("Releasing camera resource.")
        if 'cap' in locals() and cap is not None and cap.isOpened():
            cap.release()

        if args.visualize_trajectory:
            logging.info("Cleaning up visualization.")
            if vis_fig is not None: # Check if figure exists
                 try:
                     if plt.fignum_exists(vis_fig.number):
                          plt.close(vis_fig)
                 except Exception as e:
                      logging.warning(f"Error closing plot window: {e}")
            plt.ioff() # Ensure interactive mode is off

        # --- Release Robot ---
        if mc:
            logging.info("Releasing robot servos.")
            try:
                mc.release_all_servos()
                # Optionally close the serial connection if the library requires it
                # Check if the MyCobot object has a 'close' method or similar if needed
                # if hasattr(mc, 'close') and callable(mc.close): mc.close()
            except Exception as e:
                 logging.error(f"Error releasing robot servos: {e}")
        # --- End Release Robot ---

        logging.info("Live inference finished.")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Run Live Inference with Camera and MyCobot Control (Fixed Port: {ROBOT_PORT}, Fixed Baud: {ROBOT_BAUD})")

    # Required Argument
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint (.pth file)')

    # Inference Control Arguments
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use if available')
    parser.add_argument('--camera_id', type=int, default=0, help='ID of the camera to use (e.g., 0 for /dev/video0)')
    parser.add_argument('--visualize_trajectory', action='store_true', help='If set, display live camera feed during inference')
    # parser.add_argument('--min_loop_duration', type=float, default=0.1, help='Minimum time (seconds) for each control loop') # Optional rate limiting

    # Robot Control Arguments
    # --robot_port argument removed
    # --robot_baud argument removed
    parser.add_argument('--robot_speed', type=int, default=80, choices=range(0, 101), metavar="[0-100]", help='Speed for MyCobot movements (0-100)')

    # Optional: Fallback arguments if checkpoint doesn't contain 'args'
    parser.add_argument('--state_dim', type=int, default=7, help='Dimension of the state vector (angles + gripper) (if not in checkpoint)')
    parser.add_argument('--image_size', type=int, default=224, help='Image size used during training (if not in checkpoint)')
    parser.add_argument('--image_feature_dim', type=int, default=512, help='Image feature dimension (if not in checkpoint)')
    parser.add_argument('--time_emb_dim', type=int, default=64, help='Timestep embedding dimension (if not in checkpoint)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='MLP hidden dimension (if not in checkpoint)')
    parser.add_argument('--num_mlp_layers', type=int, default=4, help='Number of MLP layers (if not in checkpoint)')
    parser.add_argument('--diffusion_timesteps', type=int, default=1000, help='Number of diffusion timesteps (if not in checkpoint)')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Beta start value (if not in checkpoint)')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta end value (if not in checkpoint)')
    parser.add_argument('--use_pretrained_resnet', type=bool, default=True, help='Whether ResNet used pretrained weights (if not in checkpoint)')
    parser.add_argument('--freeze_resnet', type=bool, default=True, help='Whether ResNet weights were frozen (if not in checkpoint)')

    args = parser.parse_args()

    # --- Setup Robot Connection ---
    # Port and Baud are now hardcoded using constants defined above
    robot_mc = setup_robot(port=ROBOT_PORT, baud=ROBOT_BAUD)
    if robot_mc is None:
        logging.error(f"Failed to initialize robot connection on {ROBOT_PORT} @ {ROBOT_BAUD} baud. Exiting.")
        exit(1)
    # --- End Setup Robot Connection ---

    # Run the main inference and control loop
    run_live_inference(args, robot_mc)
