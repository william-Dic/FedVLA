import socket
import struct
import pickle
import argparse
import time

from serial.tools import list_ports
from pymycobot.mycobot import MyCobot

def recvall(sock: socket.socket, n: int) -> bytes:
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def recv_msg(sock: socket.socket) -> dict:
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

    # Connect to inference server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"Connected to server at {args.host}:{args.port}")

    try:
        while True:
            # Receive predicted action from server
            reply = recv_msg(sock)
            if reply is None:
                print("Server disconnected.")
                break

            action = reply.get('action', [])
            if len(action) < 7:
                print(f"Malformed action packet: {action}")
                time.sleep(args.interval)
                continue

            # Split into joint angles and gripper value
            pred_angles = action[:6]
            pred_gripper = action[6]

            # Send to MyCobot
            mc.send_angles(pred_angles, 50)
            mc.set_gripper_value(pred_gripper, 80)
            print(f"Executed action: angles={pred_angles}, gripper={pred_gripper}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        sock.close()
        mc.release_all_servos()
        print("Cleaned up and exited.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Inference Client for MyCobot with Diffusion Service"
    )
    parser.add_argument('--host', type=str, default='192.168.1.133',
                        help='Inference server IP')
    parser.add_argument('--port', type=int, default=46272,
                        help='Inference server port (must match server)')
    parser.add_argument('--serial_port', type=str, default=None,
                        help='Serial port for MyCobot (e.g., COM3 or /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=1000000,
                        help='Baud rate for MyCobot')
    parser.add_argument('--interval', type=float, default=0.1,
                        help='Seconds between action executions')
    args = parser.parse_args()
    main(args)
