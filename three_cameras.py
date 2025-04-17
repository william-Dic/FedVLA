import time
import os
import sys
import termios
import tty
import threading
import json
import serial
import serial.tools.list_ports
import numpy as np
import cv2  # For camera capture and saving images as PNG files

from pymycobot.mycobot import MyCobot


port: str
mc: MyCobot
sp: int = 80


def setup():
    print("")
    global port, mc
    plist = list(serial.tools.list_ports.comports())
    idx = 1
    for port in plist:
        print("{} : {}".format(idx, port))
        idx += 1

    _in = input("\nPlease input 1 - {} to choice:".format(idx - 1))
    port = str(plist[int(_in) - 1]).split(" - ")[0].strip()
    print(port)
    print("")

    baud = 1000000
    _baud = input("Please input baud(default:115200):")
    try:
        baud = int(_baud)
    except Exception:
        pass
    print(baud)
    print("")

    DEBUG = False
    f = input("Wether DEBUG mode[Y/n]:")
    if f in ["y", "Y", "yes", "Yes"]:
        DEBUG = True
    # mc = MyCobot(port, debug=True)
    mc = MyCobot(port, baud, debug=DEBUG)


class Raw(object):
    """Set raw input mode for device"""

    def __init__(self, stream):
        self.stream = stream
        self.fd = self.stream.fileno()

    def __enter__(self):
        self.original_stty = termios.tcgetattr(self.stream)
        tty.setcbreak(self.stream)

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(self.stream, termios.TCSANOW, self.original_stty)


class Helper(object):
    def __init__(self) -> None:
        self.w, self.h = os.get_terminal_size()

    def echo(self, msg):
        print("\r{}".format(" " * self.w), end="")  # Clean the screen
        print("\r{}".format(msg), end="")


class TeachingTest(Helper):
    def __init__(self, mycobot) -> None:
        super().__init__()
        self.mc = mycobot
        self.recording = False
        self.playing = False
        self.record_list = []
        self.record_gripper_list = []
        self.record_t = None
        self.play_t = None
        self.save_path = "/home/er/Desktop/FedVLA/stack_orange"  # The folder where the dataset will be stored
        self.episode_count = 1  # Track episodes
        
        # Initialize OpenCV cameras
        self.cameras = []
        self.camera_indices = [0, 1, 2]  # Camera indices for 3 views
        
        for idx in self.camera_indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                self.cameras.append(cap)
                self.echo(f"Camera {idx} initialized successfully")
            else:
                self.echo(f"Failed to initialize camera {idx}")
        
        # If not all cameras are available, fallback to available ones
        if not self.cameras:
            self.echo("No cameras found. Initializing default camera.")
            self.cameras.append(cv2.VideoCapture(0))

    def home(self):
        # Convert encoder values [122, 2053, 1929, 1900, 928, 2470] to equivalent angles
        # Using the reference material's encoder-to-angle mapping concept
        self.mc.send_angles([0,0,0,0,0,0], 80)
        self.mc.set_gripper_value(90, 80)

    def record(self):
        self.record_list = []
        self.record_gripper_list = []
        self.recording = True
        self.mc.set_fresh_mode(0)

        def _record():
            start_t = time.time()
            while self.recording:
                angles = self.mc.get_angles()
                gripper_value = self.mc.get_gripper_value()
                if angles:
                    self.record_list.append(angles)
                    self.record_gripper_list.append([gripper_value])
                    time.sleep(0.1)
                    print("\r angles{}".format(angles), end="\n")
                    print("\r gripper_value{}".format(gripper_value), end="\n")

        self.echo("Start recording.")
        self.record_t = threading.Thread(target=_record, daemon=True)
        self.record_t.start()

    def stop_record(self):
        if self.recording:
            self.recording = False
            self.record_t.join()
            self.echo("Stop record")

    def play(self):
        self.echo("Start play")
        
        self.echo("Warming up the cameras....")
        
        # Warm up all cameras
        for _ in range(6):
            for cam in self.cameras:
                ret, _ = cam.read()
            time.sleep(0.1)
        
        # Ask if the trajectory should be saved with default "n" if empty input
        save_trajectory = input("Do you want to save this trajectory? (y/n) [default n]: ").strip().lower()
        if not save_trajectory:
            save_trajectory = "n"
        
        if save_trajectory == "y":
            # Count existing episode folders in the save_path directory
            existing_episodes = [f for f in os.listdir(self.save_path) if os.path.isdir(os.path.join(self.save_path, f))]
            self.episode_count = len(existing_episodes) + 1  # New episode number
            
            # Create the new episode folder and subfolders
            episode_folder = os.path.join(self.save_path, f"episode{self.episode_count}")
            os.makedirs(episode_folder, exist_ok=True)
            
            # Create separate folders for each camera view
            frame_dirs = []
            for i in range(len(self.cameras)):
                frame_dir = os.path.join(episode_folder, f"camera{i+1}_frames")
                os.makedirs(frame_dir, exist_ok=True)
                frame_dirs.append(frame_dir)
                
            state_file = os.path.join(episode_folder, "state.json")
            state_data = []

            # Continuously capture and save frames with OpenCV while executing movements
            for i, (angles, gripper_value) in enumerate(zip(self.record_list, self.record_gripper_list)):
                # Move the robot
                self.mc.send_angles(angles, 80)
                if gripper_value[0] is None:
                    gripper_value[0] = 50
                self.mc.set_gripper_value(gripper_value[0] - 20, 80)
                
                # Dictionary to store image paths for each camera
                frame_paths = {}
                
                # Capture frames from all cameras
                for cam_idx, cam in enumerate(self.cameras):
                    ret, frame = cam.read()
                    
                    if ret:
                        # Display the captured frame (only show the first camera feed to avoid window clutter)
                        if cam_idx == 0:
                            cv2.imshow('Camera Feed', frame)
                            cv2.waitKey(1)  # Update the image window and listen for key press
                        
                        # Save the frame as an image in its corresponding folder
                        img_filename = os.path.join(frame_dirs[cam_idx], f"image{i+1}.png")
                        cv2.imwrite(img_filename, frame)
                        
                        # Store relative path to the image
                        frame_paths[f"camera{cam_idx+1}"] = f"camera{cam_idx+1}_frames/image{i+1}.png"
                
                # Save the state (angles, gripper values, and image paths)
                state_entry = {
                    "angles": angles,
                    "gripper_value": gripper_value,
                    "images": frame_paths
                }
                state_data.append(state_entry)
                
                time.sleep(0.1)
            
            # Save state to JSON file
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.echo(f"Saved trajectory to {episode_folder}")
        else:
            # Just play without saving if user chose not to save
            for angles, gripper_value in zip(self.record_list, self.record_gripper_list):
                self.mc.send_angles(angles, 80)
                if gripper_value[0] is None:
                    gripper_value[0] = 50
                self.mc.set_gripper_value(gripper_value[0] - 20, 80)
                time.sleep(0.1)
        
        self.echo("Finish play")

    def loop_play(self):
        self.playing = True

        def _loop():
            len_ = len(self.record_list)
            i = 0
            while self.playing:
                idx_ = i % len_
                i += 1
                self.mc.send_angles(self.record_list[idx_], 80)
                time.sleep(0.1)

        self.echo("Start loop play.")
        self.play_t = threading.Thread(target=_loop, daemon=True)
        self.play_t.start()

    def stop_loop_play(self):
        if self.playing:
            self.playing = False
            self.play_t.join()
            self.echo("Stop loop play.")

    def save_to_local(self):
        if not self.record_list:
            self.echo("No data should save.")
            return

        with open(os.path.dirname(__file__) + "/record.txt", "a") as f:
            json.dump(self.record_list, f, indent=2)
            self.echo(f"Saved to: {os.path.dirname(__file__)}")

    def load_from_local(self):
        with open(os.path.dirname(__file__) + "/record.txt", "r") as f:
            try:
                data = json.load(f)
                self.record_list = data
                self.echo("Load data success.")
            except Exception:
                self.echo("Error: invalid data.")
                
    def cleanup(self):
        # Release all cameras when done
        for cam in self.cameras:
            cam.release()
        cv2.destroyAllWindows()

    def print_menu(self):
        print(
            """\
        \r q: quit
        \r r: start record
        \r c: stop record
        \r p: play once
        \r P: loop play / stop loop play
        \r s: save to local
        \r l: load from local
        \r f: release mycobot
        \r h: home mycobot
        \r----------------------------------
            """
        )

    def start(self):
        self.print_menu()

        try:
            while not False:
                with Raw(sys.stdin):
                    key = sys.stdin.read(1)
                    if key == "q":
                        break
                    elif key == "r":  # Start recording
                        self.record()
                    elif key == "c":  # Stop recording
                        self.stop_record()
                    elif key == "p":  # Play and possibly save the trajectory
                        self.home()
                        self.play()
                    elif key == "P":  # Loop play
                        if not self.playing:
                            self.loop_play()
                        else:
                            self.stop_loop_play()
                    elif key == "s":  # Save to local
                        self.save_to_local()
                    elif key == "l":  # Load from local
                        self.load_from_local()
                    elif key == "f":  # Release all servos
                        self.mc.release_all_servos()
                        self.echo("Released")
                    elif key == "h":  # Home position
                        self.home()
                    else:
                        continue
        finally:
            # Make sure we clean up properly
            self.cleanup()


if __name__ == "__main__":
    setup()
    recorder = TeachingTest(mc)
    recorder.start()
