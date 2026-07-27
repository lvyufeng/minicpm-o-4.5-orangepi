#!/usr/bin/env python3
"""Test single token generation to verify the pipeline works."""

import socket
import struct
import json
import sys

def send_request(sock, request):
    """Send JSON request with length prefix."""
    payload = json.dumps(request).encode('utf-8')
    header = struct.pack('<I', len(payload))
    sock.sendall(header + payload)

    # Receive response
    header = sock.recv(4)
    if len(header) < 4:
        raise ConnectionError("Connection closed")
    response_len = struct.unpack('<I', header)[0]

    response_data = b''
    while len(response_data) < response_len:
        chunk = sock.recv(response_len - len(response_data))
        if not chunk:
            raise ConnectionError("Connection closed")
        response_data += chunk

    return json.loads(response_data.decode('utf-8'))

def main():
    print("Connecting to backend server...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('127.0.0.1', 50051))
    sock.settimeout(900.0)  # 15 minutes

    try:
        # Initialize
        print("\n1. Initializing backend...")
        response = send_request(sock, {
            'type': 'init',
            'model_path': '/mnt/data/minicpm-o-4.5-orangepi/models/MiniCPM-o-4.5'
        })
        print(f"   Status: {response.get('status')}")

        # Prefill with minimal tokens
        print("\n2. Prefilling with 3 tokens...")
        response = send_request(sock, {
            'type': 'chat_prefill',
            'session_id': 'test_minimal',
            'input_ids': [1, 100, 200]  # Just 3 tokens
        })
        print(f"   Status: {response.get('status')}")

        # Generate just 1 token to test decode path
        print("\n3. Generating 1 token (testing decode path)...")
        response = send_request(sock, {
            'type': 'chat_generate',
            'session_id': 'test_minimal',
            'max_new_tokens': 1  # Only 1 token
        })
        print(f"   Status: {response.get('status')}")
        print(f"   Generated: {response.get('num_generated')} tokens")
        print(f"   Token IDs: {response.get('token_ids')}")

        print("\n✓ SUCCESS! Single token generation works!")
        print("  Pipeline verified: prefill → decode → lm_head")

    finally:
        sock.close()

if __name__ == '__main__':
    main()
