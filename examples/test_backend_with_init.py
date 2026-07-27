#!/usr/bin/env python3
"""Test backend with init + prefill (no tokenizer needed)."""

import socket
import struct
import json
import time


def send_request(sock, request):
    """Send request and receive response."""
    payload = json.dumps(request).encode("utf-8")
    header = struct.pack("<I", len(payload))
    sock.sendall(header + payload)

    # Receive response
    header_data = b""
    while len(header_data) < 4:
        chunk = sock.recv(4 - len(header_data))
        if not chunk:
            raise ConnectionError("Connection closed")
        header_data += chunk

    response_len = struct.unpack("<I", header_data)[0]

    response_data = b""
    while len(response_data) < response_len:
        chunk = sock.recv(response_len - len(response_data))
        if not chunk:
            raise ConnectionError("Connection closed")
        response_data += chunk

    return json.loads(response_data.decode("utf-8"))


def test_backend():
    """Test backend with init + prefill."""
    print("=" * 60)
    print("Testing Backend: Init + Prefill")
    print("=" * 60)

    # Connect to backend
    print("\n[1/4] Connecting to backend server...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 50051))
    sock.settimeout(900.0)  # 15 min for init
    print("✓ Connected")

    # Initialize model
    print("\n[2/4] Initializing model (may take several minutes)...")
    start = time.time()
    response = send_request(sock, {
        "type": "init",
        "request_id": 0,
        "model_path": "/mnt/data/minicpm-o-4.5-orangepi/models/MiniCPM-o-4.5",
    })
    elapsed = time.time() - start

    if response.get("status") == "ok":
        print(f"✓ Model initialized in {elapsed:.1f}s")
    else:
        print(f"✗ Init failed: {response.get('error')}")
        sock.close()
        return

    # Check metrics
    print("\n[3/4] Checking metrics...")
    sock.settimeout(30.0)
    response = send_request(sock, {"type": "metrics", "request_id": 1})
    print(f"  Metrics: {response.get('metrics', {})}")

    # Test prefill with sample tokens
    print("\n[4/4] Testing prefill with sample tokens...")
    # Simple token sequence: assume token IDs 1000-1005 are valid
    input_ids = [1, 1000, 1001, 1002, 1003, 1004, 1005]

    response = send_request(sock, {
        "type": "chat_prefill",
        "request_id": 2,
        "session_id": "test_session_1",
        "input_ids": input_ids,
    })

    print(f"  Response status: {response.get('status')}")
    if response.get("status") == "ok":
        print(f"✓ Prefill succeeded!")
        print(f"  KV cache length: {response.get('kv_cache_length', 'N/A')}")
    else:
        print(f"✗ Prefill failed: {response.get('error')}")

    sock.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_backend()
