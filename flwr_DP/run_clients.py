import subprocess
import sys
import time
import os

def run_client(client_id, num_clients):
    """Run a single client process."""
    cmd = [
        sys.executable,
        "flwr_client.py",
        "--client-id", str(client_id),
        "--num-clients", str(num_clients)
    ]
    return subprocess.Popen(cmd)

def main():
    num_clients = 3
    processes = []

    print(f"Starting {num_clients} clients...")
    
    # Start all clients
    for client_id in range(num_clients):
        print(f"Starting client {client_id}...")
        process = run_client(client_id, num_clients)
        processes.append(process)
        time.sleep(2)  # Give each client time to initialize

    try:
        # Wait for all clients to finish
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        print("\nShutting down clients...")
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait()
        print("All clients terminated.")

if __name__ == "__main__":
    main() 