# client.py
import socket
import time

SERVER_IP = '10.231.112.52'  # <-- 你的 Windows 服务器在 WLAN 网卡上的地址
PORT = 50007

def run_client():
    time.sleep(1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"尝试连接到服务器 {SERVER_IP}:{PORT} …")
        s.connect((SERVER_IP, PORT))
        message = "Hello, Robot Server!"
        s.sendall(message.encode('utf-8'))
        print(f"已发送：{message}")
        data = s.recv(1024)
        print(f"收到服务器回执：{data.decode('utf-8')}")
        print("客户端结束")

if __name__ == "__main__":
    run_client()
