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
import cv2  # For saving the images as PNG files
import pyrealsense2 as rs  # For RealSense

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
        self.save_path = "stack_orange"  # The folder where the dataset will be stored
        self.episode_count = 1  # Track episodes
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

    def home(self):
        self.mc.set_encoders([122, 2053, 1929, 1900, 928, 2470], 80)
        self.mc.set_gripper_value(90, 80)

    def record(self):
        self.record_list = []
        self.record_gripper_list = []
        self.recording = True
        self.mc.set_fresh_mode(0)

        def _record():
            start_t = time.time()
            while self.recording:
                angles = self.mc.get_encoders()
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
            frame_dir = os.path.join(episode_folder, "frame_dir")
            os.makedirs(frame_dir, exist_ok=True)
            state_file = os.path.join(episode_folder, "state.json")
            state_data = []
    
            # Capture and save frames with RealSense
            for i, (angles, gripper_value) in enumerate(zip(self.record_list, self.record_gripper_list)):
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
    
                if not color_frame:
                    continue
    
                # Convert color frame to numpy array
                color_image = np.asanyarray(color_frame.get_data())
    
                # Display the captured frame
                cv2.imshow('RealSense Camera Feed', color_image)
                cv2.waitKey(1)  # Update the image window and listen for key press (frame update)

                # Save each frame as an image
                img_filename = os.path.join(frame_dir, f"image{i+1}.png")
                cv2.imwrite(img_filename, color_image)
    
                # Save the state (angles and gripper values)
                state_data.append({
                    "angles": angles,
                    "gripper_value": gripper_value
                })
    
            # Save state to JSON file
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
    
            self.echo(f"Saved trajectory to {episode_folder}")

        # Continue the play process (move the robot based on the recorded trajectory)
        for angles, gripper_value in zip(self.record_list, self.record_gripper_list):
            self.mc.set_encoders(angles, 80)
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
                self.mc.set_encoders(self.record_list[idx_], 80)
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


if __name__ == "__main__":
    setup()
    recorder = TeachingTest(mc)
    recorder.start()
