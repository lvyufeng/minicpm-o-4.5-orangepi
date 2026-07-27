#!/usr/bin/env python3
"""Test backend directly with pre-tokenized input (bypass tokenizer issue)."""

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
    """Test backend with hardcoded token IDs."""
    print("=" * 60)
    print("Testing Backend Directly (No Tokenizer)")
    print("=" * 60)

    # Connect to backend
    print("\n[1/3] Connecting to backend server...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 50051))
    sock.settimeout(30.0)
    print("✓ Connected")

    # Check if already initialized
    print("\n[2/3] Getting metrics...")
    response = send_request(sock, {"type": "metrics", "request_id": 0})
    print(f"Response: {response}")

    # Test prefill with hardcoded tokens: "Hello world"
    # Using token IDs that should work for most models
    # Typically: [BOS, token1, token2, ...]
    print("\n[3/3] Testing prefill with sample tokens...")
    input_ids = [1, 9906, 1879]  # Common BPE tokens for "Hello world"

    response = send_request(sock, {
        "type": "chat_prefill",
        "request_id": 1,
        "session_id": "test_session_1",
        "input_ids": input_ids,
    })

    print(f"Prefill response: {response}")

    if response.get("status") == "ok":
        print("✓ Prefill succeeded!")
        print(f"  KV cache length: {response.get('kv_cache_length', 'unknown')}")
    else:
        print(f"✗ Prefill failed: {response.get('error')}")

    sock.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_backend()
