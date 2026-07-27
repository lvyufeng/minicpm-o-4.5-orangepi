#!/usr/bin/env python3
"""Benchmark text generation performance."""

import socket
import struct
import json
import time
import sys

def send_request(sock, request):
    """Send JSON request and receive response."""
    payload = json.dumps(request).encode('utf-8')
    header = struct.pack('<I', len(payload))
    sock.sendall(header + payload)

    resp_header = sock.recv(4)
    resp_len = struct.unpack('<I', resp_header)[0]
    resp_data = b''
    while len(resp_data) < resp_len:
        chunk = sock.recv(resp_len - len(resp_data))
        if not chunk:
            raise ConnectionError("Connection closed")
        resp_data += chunk

    return json.loads(resp_data.decode('utf-8'))

def main():
    print("Connecting to backend server...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('127.0.0.1', 50051))
    sock.settimeout(900)

    try:
        # Initialize
        print("\n1. Initializing backend...")
        start = time.time()
        response = send_request(sock, {
            "type": "init",
            "model_path": "/mnt/data/minicpm-o-4.5-orangepi/models/MiniCPM-o-4.5"
        })
        init_time = time.time() - start
        print(f"   Init time: {init_time:.2f}s")
        print(f"   Status: {response.get('status')}")

        # Prefill with longer sequence
        print("\n2. Prefilling with 10 tokens...")
        start = time.time()
        response = send_request(sock, {
            "type": "chat_prefill",
            "session_id": "bench",
            "input_ids": list(range(1, 11))  # [1, 2, ..., 10]
        })
        prefill_time = time.time() - start
        print(f"   Prefill time: {prefill_time:.2f}s ({10/prefill_time:.2f} tokens/s)")
        print(f"   Status: {response.get('status')}")

        # Generate tokens
        print("\n3. Generating 10 tokens...")
        start = time.time()
        response = send_request(sock, {
            "type": "chat_generate",
            "session_id": "bench",
            "max_new_tokens": 10,
            "generate_audio": False
        })
        gen_time = time.time() - start

        result = response.get('result', {})
        num_tokens = result.get('num_generated', 0)
        tokens_per_sec = num_tokens / gen_time if gen_time > 0 else 0

        print(f"   Generation time: {gen_time:.2f}s")
        print(f"   Tokens generated: {num_tokens}")
        print(f"   Speed: {tokens_per_sec:.2f} tokens/s")
        print(f"   Token IDs: {result.get('token_ids')}")

        print(f"\n{'='*50}")
        print(f"SUMMARY:")
        print(f"  Prefill:  {10/prefill_time:.2f} tokens/s")
        print(f"  Decode:   {tokens_per_sec:.2f} tokens/s")
        print(f"{'='*50}")

    finally:
        sock.close()

if __name__ == '__main__':
    main()
