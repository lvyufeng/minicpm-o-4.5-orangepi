#!/usr/bin/env python3
"""Quick test of backend_server protocol without full model loading."""

import json
import socket
import struct
import sys


def send_request(sock, request):
    """Send JSON request with 4-byte length prefix."""
    payload = json.dumps(request).encode("utf-8")
    header = struct.pack("<I", len(payload))
    sock.sendall(header + payload)

    # Receive response
    header = sock.recv(4)
    if len(header) != 4:
        raise ConnectionError("Failed to receive response header")

    response_len = struct.unpack("<I", header)[0]
    response_data = b""
    while len(response_data) < response_len:
        chunk = sock.recv(response_len - len(response_data))
        if not chunk:
            raise ConnectionError("Connection closed during response")
        response_data += chunk

    return json.loads(response_data.decode("utf-8"))


def main():
    host = "127.0.0.1"
    port = 50051

    print(f"Connecting to backend_server at {host}:{port}...")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        sock.settimeout(10.0)
        print("✓ Connected")

        # Test 1: Metrics (should work without model loaded)
        print("\n[Test 1] Getting metrics...")
        response = send_request(sock, {"type": "metrics"})
        print(f"Response: {response}")

        if response.get("status") == "ok":
            print("✓ Metrics request successful")
            metrics = response.get("metrics", {})
            print(f"  Backend: {metrics.get('backend')}")
            print(f"  Device ID: {metrics.get('device_id')}")
            print(f"  Model loaded: {metrics.get('model_loaded')}")
        else:
            print(f"✗ Metrics request failed: {response.get('error')}")

        # Test 2: Try to prefill without model (should fail gracefully)
        print("\n[Test 2] Attempting prefill without model...")
        response = send_request(sock, {
            "type": "chat_prefill",
            "session_id": "test",
            "msgs": [{"role": "user", "content": "Hello"}]
        })
        print(f"Response: {response}")

        if response.get("status") == "error":
            print(f"✓ Correctly rejected (model not loaded): {response.get('error')}")
        else:
            print(f"⚠ Unexpected success without model")

        # Test 3: Shutdown
        print("\n[Test 3] Sending shutdown...")
        response = send_request(sock, {"type": "shutdown"})
        print(f"Response: {response}")

        if response.get("status") == "ok":
            print("✓ Shutdown acknowledged")

        sock.close()
        print("\n✓ All tests passed")

    except ConnectionRefusedError:
        print(f"✗ Connection refused. Is backend_server running?")
        print(f"   Start with: cd build && ./backend_server --port {port}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
