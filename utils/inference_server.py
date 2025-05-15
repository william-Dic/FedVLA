# server.py
import socket

HOST = ''       # 监听所有可用网卡
PORT = 50007    # 自定义端口，确保未被占用

def run_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"服务器启动，监听 {HOST or '0.0.0.0'}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print(f"已连接：{addr}")
            data = conn.recv(1024)
            if data:
                text = data.decode('utf-8')
                print(f"收到客户端消息：{text}")
                reply = "Message received: " + text
                conn.sendall(reply.encode('utf-8'))
                print("已发送回执给客户端")
        print("连接已关闭")

if __name__ == "__main__":
    run_server()
