import socket
import pickle
import struct
import base64
import argparse
import time

# Optional: use OpenCV for camera capture; install opencv-python if needed
try:
    import cv2
except ImportError:
    cv2 = None

from PIL import Image
import io

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


def get_robot_state() -> (list, float):
    """
    Stub: replace with actual robot API to fetch angles and gripper value.
    Return: (angles list of floats, gripper float)
    """
    # Example placeholder values
    angles = [0.0] * 7
    gripper = 0.0
    return angles, gripper


def capture_image(image_path: str = None) -> bytes:
    """
    Capture an image either from file or camera.
    If image_path is provided, load from disk; otherwise use camera (if cv2 available).
    Returns raw image bytes (JPEG).
    """
    if image_path:
        with open(image_path, 'rb') as f:
            return f.read()
    if cv2:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to capture image from camera")
        _, buf = cv2.imencode('.jpg', frame)
        return buf.tobytes()
    raise RuntimeError("No image source available. Provide --image_path or install OpenCV.")


def main(args):
    # Connect to server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"Connected to server at {args.host}:{args.port}")

    try:
        while True:
            # 1. Fetch robot state
            angles, gripper_val = get_robot_state()
            # 2. Capture image
            img_bytes = capture_image(args.image_path)
            img_b64 = base64.b64encode(img_bytes).decode('ascii')

            # 3. Build message dict
            msg = {
                'angles': angles,
                'gripper_value': [gripper_val],
                'image': img_b64
            }

            # 4. Send to server
            send_msg(sock, msg)
            print("Sent state and image to server.")

            # 5. Receive action
            reply = recv_msg(sock)
            if reply is None:
                print("Server disconnected.")
                break
            action = reply.get('action', [])
            print(f"Received action: {action}")

            # 6. Wait before next inference
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("Interrupted by user, exiting.")
    finally:
        sock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Client for Diffusion Policy Inference Service")
    parser.add_argument('--host', type=str, default='10.231.112.52', help='Server IP address')
    parser.add_argument('--port', type=int, default=50007, help='Server port')
    parser.add_argument('--interval', type=float, default=1.0, help='Seconds between inferences')
    parser.add_argument('--image_path', type=str, default=None, help='Optional path to image file')
    args = parser.parse_args()
    main(args)
