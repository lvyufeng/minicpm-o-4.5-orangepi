#!/usr/bin/env python3
"""Test text generation with the Orange Pi backend."""

import socket
import struct
import json
import sys

def send_request(sock, request):
    """Send JSON request and receive response."""
    payload = json.dumps(request).encode('utf-8')
    header = struct.pack('<I', len(payload))
    sock.sendall(header + payload)

    # Receive response
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
    sock.settimeout(900)  # 15 min for model loading

    try:
        # 1. Initialize
        print("\n1. Initializing backend...")
        response = send_request(sock, {
            "type": "init",
            "model_path": "/mnt/data/minicpm-o-4.5-orangepi/models/MiniCPM-o-4.5"
        })
        print(f"   Status: {response.get('status')}")

        # 2. Prefill with some tokens
        print("\n2. Prefilling with tokens [1, 100, 200, 300]...")
        response = send_request(sock, {
            "type": "chat_prefill",
            "session_id": "test_gen",
            "input_ids": [1, 100, 200, 300]
        })
        print(f"   Status: {response.get('status')}")

        # 3. Generate 5 tokens
        print("\n3. Generating 5 tokens...")
        response = send_request(sock, {
            "type": "chat_generate",
            "session_id": "test_gen",
            "max_new_tokens": 5,
            "generate_audio": False
        })

        print(f"   Status: {response.get('status')}")
        result = response.get('result', {})
        print(f"   Generated: {result.get('num_generated')} tokens")
        print(f"   Token IDs: {result.get('token_ids')}")
        print(f"   Finished: {result.get('finished')}")

        if result.get('num_generated') == 5:
            print("\n✓ SUCCESS! Generation works correctly!")
        else:
            print(f"\n✗ FAILED! Expected 5 tokens, got {result.get('num_generated')}")
            sys.exit(1)

    finally:
        sock.close()

if __name__ == '__main__':
    main()
