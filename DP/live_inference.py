# FedVLA/DP/live_inference.py (Workflow: Release->Home->Wait->Loop->Home)
# Modified to use reduced inference_timesteps for CPU speedup

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm # Can be removed if loop isn't long/doesn't need progress bar
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
import serial.tools.list_ports
from pymycobot.mycobot import MyCobot
# --- End Robot Control Imports ---

# Import custom modules
try:
    from model import DiffusionPolicyModel
    # Ensure p_sample_loop can handle varying timesteps or uses a method compatible with step reduction
    from train import linear_beta_schedule, extract, p_sample, p_sample_loop
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure model.py and train.py (or the necessary functions) are accessible.")
    print("You may also need 'opencv-python', 'pyserial', 'pymycobot':")
    print("pip install opencv-python pyserial pymycobot")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]' # Less relevant now

# --- Fixed Robot Parameters ---
ROBOT_PORT = "/dev/ttyAMA0"
ROBOT_BAUD = 1000000
HOME_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Example home angles
HOME_GRIPPER = 90 # Example home gripper value (adjust as needed)
HOMING_SPEED = 50 # Speed for homing movements
HOMING_WAIT_TIME = 5 # Seconds to wait after sending homing command
INITIAL_WAIT_TIME = 3 # Seconds to wait after homing before starting loop
# --- End Fixed Robot Parameters ---

# --- Robot Setup Function ---
def setup_robot(port: str, baud: int = ROBOT_BAUD) -> Optional[MyCobot]:
    """Initializes the MyCobot connection on the specified port and baud rate."""
    global mc
    mc = None
    logging.info(f"Attempting to connect to MyCobot on fixed port: {port} at fixed baud: {baud}.")
    try:
        mc = MyCobot(port, baudrate=baud)
        time.sleep(0.5) # Allow connection to establish
        angles = mc.get_angles()
        if not isinstance(angles, list) or len(angles) != 6:
             time.sleep(1.0) # Longer delay and retry
             angles = mc.get_angles()
             if not isinstance(angles, list) or len(angles) != 6:
                 logging.error(f"Failed to get valid initial angles from robot on {port} after retries. Check connection/power.")
                 return None
             else:
                  logging.info("Successfully retrieved angles after longer delay.")
        logging.info(f"Successfully connected to MyCobot on {port}.")
        logging.info(f"Initial robot angles: {angles}")
        return mc
    except serial.SerialException as e:
        logging.error(f"Failed to connect to MyCobot on port {port}: {e}. Check permissions/wiring.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during robot setup on port {port}: {e}")
        return None
# --- End Robot Setup Function ---


def run_live_inference(args, mc: MyCobot):
    """
    Loads model, connects to robot/camera, and executes the workflow:
    Release -> Home -> Wait -> Task Loop (Predict & Act) -> Home -> Release.
    Uses reduced inference steps for potentially faster CPU execution.
    """
    # --- Device Setup ---
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}") # Should show 'cpu' if no GPU

    # --- Load Checkpoint ---
    if not os.path.isfile(args.checkpoint_path):
        logging.error(f"Checkpoint file not found: {args.checkpoint_path}"); return
    logging.info(f"Loading checkpoint from: {args.checkpoint_path}")
    # Load to CPU explicitly if device is CPU
    map_location = device if str(device) == "cpu" else None
    checkpoint = torch.load(args.checkpoint_path, map_location=map_location)
    train_args = checkpoint.get('args')
    if train_args is None:
        logging.warning("Checkpoint missing training arguments ('args'). Using CLI args as fallback.")
        train_args = vars(args)
    elif isinstance(train_args, argparse.Namespace):
        train_args = vars(train_args)

    # --- Diffusion Schedule Setup ---
    try:
        # Get the original number of timesteps the model was trained with
        original_timesteps = train_args.get('diffusion_timesteps', args.diffusion_timesteps)
        beta_start = train_args.get('beta_start', args.beta_start)
        beta_end = train_args.get('beta_end', args.beta_end)
    except KeyError as e:
        logging.error(f"Missing critical diffusion parameter '{e}'."); return
    # Calculate schedule parameters based on original timesteps
    betas = linear_beta_schedule(timesteps=original_timesteps, beta_start=beta_start, beta_end=beta_end).to(device)
    alphas = 1. - betas; alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod); sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas); posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    logging.info(f"Diffusion schedule calculated based on original {original_timesteps} timesteps.")

    # --- Model Initialization ---
    try:
        state_dim = train_args.get('state_dim', args.state_dim)
        if state_dim != 7: logging.warning(f"State_dim is {state_dim}, expected 7.")
        time_emb_dim = train_args.get('time_emb_dim', args.time_emb_dim)
        hidden_dim = train_args.get('hidden_dim', args.hidden_dim)
        num_mlp_layers = train_args.get('num_mlp_layers', args.num_mlp_layers)
        image_feature_dim = train_args.get('image_feature_dim', args.image_feature_dim)
        use_pretrained_resnet = train_args.get('use_pretrained_resnet', args.use_pretrained_resnet)
        freeze_resnet = train_args.get('freeze_resnet', args.freeze_resnet)
    except KeyError as e:
        logging.error(f"Missing critical model parameter '{e}'."); return
    model = DiffusionPolicyModel(state_dim=state_dim, time_emb_dim=time_emb_dim, hidden_dim=hidden_dim,
                                 num_layers=num_mlp_layers, image_feature_dim=image_feature_dim,
                                 use_pretrained_resnet=use_pretrained_resnet, freeze_resnet=freeze_resnet).to(device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("Successfully loaded model state dict.")
    except Exception as e:
        logging.exception(f"Error loading state dict: {e}"); return
    model.eval()

    # --- Image Transform ---
    image_size = train_args.get('image_size', args.image_size)
    image_transform = transforms.Compose([ transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
    logging.info(f"Image transform set up for size {image_size}x{image_size}.")

    # --- Camera Setup ---
    logging.info(f"Attempting to open camera ID: {args.camera_id}")
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        logging.error(f"Error: Could not open camera with ID {args.camera_id}.")
        if cap: cap.release(); return
    logging.info("Camera opened successfully.")

    # --- Visualization Setup ---
    vis_fig, vis_ax = None, None
    if args.visualize_trajectory:
        logging.info("Visualization enabled. Press 'q' in the window or close it to quit early.")
        plt.ion(); vis_fig, vis_ax = plt.subplots(1, 1, figsize=(6, 6))
        vis_fig.suptitle("Live Camera Feed & Prediction")

    # --- START WORKFLOW ---
    try:
        # 1. Release Servos
        logging.info("Releasing robot servos...")
        mc.release_all_servos()
        time.sleep(1.0) # Short pause after release

        # 2. Go Home
        logging.info(f"Sending robot to home position: Angles={HOME_ANGLES}, Gripper={HOME_GRIPPER} at speed {HOMING_SPEED}...")
        mc.send_angles(HOME_ANGLES, HOMING_SPEED)
        mc.set_gripper_value(HOME_GRIPPER, HOMING_SPEED)
        logging.info(f"Waiting {HOMING_WAIT_TIME} seconds for robot to reach home...")
        time.sleep(HOMING_WAIT_TIME)
        logging.info("Robot should be home.")

        # 3. Wait
        logging.info(f"Waiting {INITIAL_WAIT_TIME} seconds before starting task...")
        time.sleep(INITIAL_WAIT_TIME)

        # 4. Start Control Loop
        logging.info(f"Starting control loop for a maximum of {args.max_steps} steps...")
        frame_count = 0
        task_loop_active = True
        robot_speed = args.robot_speed # Speed for task execution

        with torch.no_grad():
            while task_loop_active and frame_count < args.max_steps:
                # --- Check for Manual Stop (Visualization Window Closed) ---
                if args.visualize_trajectory and (vis_fig is None or not plt.fignum_exists(vis_fig.number)):
                     logging.info("Visualization window closed. Stopping task loop.")
                     task_loop_active = False
                     break # Exit the while loop

                # --- Capture Image ---
                ret, frame = cap.read()
                if not ret:
                    logging.error("Error: Can't receive frame (stream end?). Stopping loop.")
                    task_loop_active = False; break

                # --- Process Image ---
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                try:
                    image_tensor = image_transform(pil_image).unsqueeze(0).to(device)
                except Exception as e:
                    logging.warning(f"Skipping step {frame_count} due to image transform error: {e}.")
                    frame_count += 1 # Increment even if skipped
                    continue

                # --- Visualize (Optional) ---
                if args.visualize_trajectory and vis_ax is not None and vis_fig is not None:
                    try:
                        vis_ax.clear(); vis_ax.imshow(pil_image)
                        vis_ax.set_title(f"Step: {frame_count}/{args.max_steps} - Close window to stop")
                        vis_ax.axis('off'); plt.draw(); plt.pause(0.001)
                    except Exception as e:
                        logging.warning(f"Error during visualization update: {e}")

                # --- Predict Next State ---
                #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
                # START MODIFICATION for CPU Speedup
                # Define the number of steps for inference (much smaller than original_timesteps)
                inference_timesteps = 20 # <--- ADJUST THIS VALUE! Try 50, 20, 10, etc.
                logging.info(f"CPU inference: Attempting prediction with {inference_timesteps} steps.")

                inference_start_time = time.time() # Start timing sampling

                # Call p_sample_loop using the reduced inference_timesteps
                predicted_state_batch = p_sample_loop(
                    model,
                    shape=(1, state_dim),
                    timesteps=inference_timesteps, # Use reduced steps here
                    betas=betas,                  # Pass original schedule parameters
                    sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                    sqrt_recip_alphas=sqrt_recip_alphas,
                    posterior_variance=posterior_variance,
                    device=device,                # Should be 'cpu'
                    image_input=image_tensor
                )
                inference_time = time.time() - inference_start_time # End timing sampling
                # END MODIFICATION for CPU Speedup
                #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                # --- Extract and Send Action ---
                pred_state_np = predicted_state_batch.squeeze(0).cpu().numpy()
                if len(pred_state_np) == state_dim:
                    predicted_angles = pred_state_np[:-1].tolist()
                    predicted_gripper = pred_state_np[-1]
                    predicted_gripper_clamped = np.clip(predicted_gripper, 0, 100)

                    print(f"\n--- Step {frame_count} ---")
                    np.set_printoptions(precision=3, suppress=True)
                    print(f"  Predicted Angles:   {predicted_angles}")
                    print(f"  Predicted Gripper:  {predicted_gripper:.2f} (Clamped: {int(round(predicted_gripper_clamped))})")
                    # Log the inference time for the reduced steps
                    print(f"  Inference Time ({inference_timesteps} steps): {inference_time:.4f} seconds") # <-- Updated log message

                    try:
                        if mc:
                            mc.send_angles(predicted_angles, robot_speed)
                            mc.set_gripper_value(int(round(predicted_gripper_clamped)), robot_speed)
                        else:
                            logging.error("Robot object invalid. Cannot send command."); task_loop_active = False; break
                    except Exception as e:
                        logging.error(f"Error sending commands to robot: {e}"); task_loop_active = False; break # Stop loop on error
                else:
                    logging.warning(f"Predicted state dim ({len(pred_state_np)}) != expected ({state_dim}). Skipping step.")

                frame_count += 1
                # Optional small delay between steps if needed
                # time.sleep(0.05)

            logging.info(f"Task loop finished after {frame_count} steps.")

        # 5. Go Home Again
        logging.info(f"Task finished. Sending robot back home at speed {HOMING_SPEED}...")
        try:
             if mc:
                  mc.send_angles(HOME_ANGLES, HOMING_SPEED)
                  mc.set_gripper_value(HOME_GRIPPER, HOMING_SPEED)
                  logging.info(f"Waiting {HOMING_WAIT_TIME} seconds for robot to reach home...")
                  time.sleep(HOMING_WAIT_TIME)
                  logging.info("Robot should be home.")
             else:
                  logging.error("Robot object invalid. Cannot send home command.")
        except Exception as e:
             logging.error(f"Error sending robot home: {e}")

        # 6. Release Servos (Optional final step)
        # logging.info("Releasing servos.")
        # if mc: mc.release_all_servos()


    except KeyboardInterrupt:
        logging.info("Keyboard interrupt detected. Stopping workflow.")
        # Attempt to go home even if interrupted mid-task
        logging.warning("Attempting to send robot home after interrupt...")
        try:
             if mc:
                  mc.send_angles(HOME_ANGLES, HOMING_SPEED)
                  mc.set_gripper_value(HOME_GRIPPER, HOMING_SPEED)
                  time.sleep(HOMING_WAIT_TIME / 2) # Shorter wait
             else:
                  logging.error("Robot object invalid during interrupt handling.")
        except Exception as e:
             logging.error(f"Error sending robot home during interrupt handling: {e}")

    finally:
        # --- Cleanup ---
        logging.info("Cleaning up resources...")
        if 'cap' in locals() and cap is not None and cap.isOpened():
            cap.release()
            logging.info("Camera released.")

        if args.visualize_trajectory:
            if vis_fig is not None:
                 try: plt.close(vis_fig)
                 except Exception: pass # Ignore errors during cleanup plot closing
            plt.ioff()
            logging.info("Visualization closed.")

        if mc:
            logging.info("Releasing robot servos as final step.")
            try: mc.release_all_servos()
            except Exception as e: logging.error(f"Error releasing robot servos during final cleanup: {e}")

        logging.info("Workflow finished.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Default diffusion_timesteps here is mainly for fallback if checkpoint lacks args
    # The actual inference steps are controlled by 'inference_timesteps' inside the loop now
    parser = argparse.ArgumentParser(description=f"Run MyCobot Live Inference Workflow (Fixed Port: {ROBOT_PORT}, Fixed Baud: {ROBOT_BAUD})")

    # Required Argument
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint (.pth file)')

    # Inference Control Arguments
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (set > -1 for GPU, otherwise uses CPU)') # Keep arg, but logic uses cpu if cuda unavailable
    parser.add_argument('--camera_id', type=int, default=0, help='ID of the camera to use (e.g., 0 for /dev/video0)')
    parser.add_argument('--visualize_trajectory', action='store_true', help='If set, display live camera feed during inference')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum number of predict-and-act steps in the task loop')

    # Robot Control Arguments
    parser.add_argument('--robot_speed', type=int, default=75, choices=range(0, 101), metavar="[0-100]", help='Speed for MyCobot movements during the task (0-100)')

    # Optional: Fallback arguments if checkpoint doesn't contain 'args'
    parser.add_argument('--state_dim', type=int, default=7, help='Dimension of the state vector (if not in checkpoint)')
    parser.add_argument('--image_size', type=int, default=224, help='Image size used during training (if not in checkpoint)')
    parser.add_argument('--image_feature_dim', type=int, default=512, help='Image feature dimension (if not in checkpoint)')
    parser.add_argument('--time_emb_dim', type=int, default=64, help='Timestep embedding dimension (if not in checkpoint)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='MLP hidden dimension (if not in checkpoint)')
    parser.add_argument('--num_mlp_layers', type=int, default=4, help='Number of MLP layers (if not in checkpoint)')
    parser.add_argument('--diffusion_timesteps', type=int, default=1000, help='Original number of diffusion timesteps model trained with (if not in checkpoint)') # Clarified help text
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Beta start value (if not in checkpoint)')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta end value (if not in checkpoint)')
    parser.add_argument('--use_pretrained_resnet', type=bool, default=True, help='Whether ResNet used pretrained weights (if not in checkpoint)')
    parser.add_argument('--freeze_resnet', type=bool, default=True, help='Whether ResNet weights were frozen (if not in checkpoint)')


    args = parser.parse_args()

    # --- Setup Robot Connection ---
    robot_mc = setup_robot(port=ROBOT_PORT, baud=ROBOT_BAUD)
    if robot_mc is None:
        logging.error(f"Failed to initialize robot connection. Exiting.")
        exit(1)
    # --- End Setup Robot Connection ---

    # Run the main workflow
    run_live_inference(args, robot_mc)
