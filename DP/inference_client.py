import socket
import pickle
import struct
import base64
import argparse
import time
import io

import cv2
from PIL import Image

from serial.tools import list_ports
from pymycobot.mycobot import MyCobot


def send_msg(sock: socket.socket, msg: object) -> None:
    """
    Send a Python object via pickle with a 4-byte length prefix.
    """
    data = pickle.dumps(msg)
    length = struct.pack('>I', len(data))
    sock.sendall(length + data)


def recvall(sock: socket.socket, n: int) -> bytes:
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_msg(sock: socket.socket) -> object:
    raw_len = recvall(sock, 4)
    if raw_len is None:
        return None
    msg_len = struct.unpack('>I', raw_len)[0]
    data = recvall(sock, msg_len)
    return pickle.loads(data)


def find_mycobot_port() -> str:
    ports = list_ports.comports()
    for p in ports:
        if 'ttyUSB' in p.device or 'COM' in p.device:
            return p.device
    raise RuntimeError("No MyCobot serial port found. Specify with --serial_port.")


def main(args):
    # Initialize MyCobot
    serial_port = args.serial_port or find_mycobot_port()
    mc = MyCobot(serial_port, args.baud, debug=False)
    print(f"Connected to MyCobot on {serial_port} at {args.baud} baud.")

    # Home robot and gripper
    mc.send_angles([0, 0, 0, 0, 0, 0], 50)
    mc.set_gripper_value(90, 80)
    print("Robot and gripper initialized to home position.")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera.")
    # Warm up camera
    for _ in range(6):
        ret, _ = cap.read()
        time.sleep(0.1)
    print("Camera warmed up and ready.")

    # Connect to inference server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"Connected to server at {args.host}:{args.port}")

    try:
        while True:
            # Read robot state
            angles = mc.get_angles() or [0.0] * 7
            gripper_val = mc.get_gripper_value() or 0.0

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image. Skipping.")
                time.sleep(args.interval)
                continue

            # Encode image to JPEG bytes
            _, buf = cv2.imencode('.jpg', frame)
            img_bytes = buf.tobytes()
            img_b64 = base64.b64encode(img_bytes).decode('ascii')

            # Build and send message
            msg = {
                'angles': angles,
                'gripper_value': [gripper_val],
                'image': img_b64
            }
            send_msg(sock, msg)
            print(f"Sent state: {angles} + {gripper_val} and image to server.")

            # Receive predicted action
            reply = recv_msg(sock)
            if reply is None:
                print("Server disconnected.")
                break
            action = reply.get('action', [])
            print(f"Received action: {action}")

            # Execute predicted action
            pred_angles = action[:7]
            pred_gripper = action[-1]
            mc.send_angles(pred_angles, 50)
            mc.set_gripper_value(int(pred_gripper), 80)
            print(f"Executed action: angles={pred_angles}, gripper={pred_gripper}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        sock.close()
        mc.release_all_servos()
        print("Cleaned up and exited.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference Client for MyCobot with Diffusion Service")
    parser.add_argument('--host', type=str, default='10.231.112.52', help='Inference server IP')
    parser.add_argument('--port', type=int, default=50007, help='Inference server port')
    parser.add_argument('--serial_port', type=str, default='/dev/ttyAMA0', help='Serial port for MyCobot (e.g., COM3 or /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=1000000, help='Baud rate for MyCobot')
    parser.add_argument('--interval', type=float, default=0.1, help='Seconds between inference calls')
    args = parser.parse_args()
    main(args)
