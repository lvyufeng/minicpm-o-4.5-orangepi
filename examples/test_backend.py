#!/usr/bin/env python3
"""Test Orange Pi backend connectivity and basic inference."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend import create_backend


def test_backend_connection():
    """Test connecting to backend server."""
    print("=" * 60)
    print("Testing Orange Pi Backend Connection")
    print("=" * 60)

    # Create backend (will connect to running backend_server)
    backend = create_backend(
        backend_type="orangepi",
        model_path="/path/to/model",  # Not used by client, server loads model
        gpu_id=0,
        backend_server_host="127.0.0.1",
        backend_server_port=50051,
    )

    print("\n[1/4] Connecting to backend server...")
    try:
        backend.load_model()
        print("✓ Connected successfully")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nMake sure backend_server is running:")
        print("  cd build && ./backend_server --model_path <path> --port 50051")
        return False

    print("\n[2/4] Getting metrics...")
    try:
        metrics = backend.metrics()
        print(f"✓ Backend type: {metrics.get('backend', 'unknown')}")
        print(f"  KV cache length: {metrics.get('kv_cache_length', 0)}")
    except Exception as e:
        print(f"✗ Failed to get metrics: {e}")

    print("\n[3/4] Testing chat prefill...")
    try:
        msgs = [
            {"role": "user", "content": "Hello, who are you?"}
        ]
        prompt = backend.chat_prefill(
            session_id="test-session",
            msgs=msgs,
            omni_mode=False,
        )
        print(f"✓ Prefill successful, prompt length: {len(prompt)}")
    except Exception as e:
        print(f"✗ Prefill failed: {e}")

    print("\n[4/4] Shutting down...")
    try:
        backend.shutdown()
        print("✓ Shutdown successful")
    except Exception as e:
        print(f"✗ Shutdown failed: {e}")

    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)
    return True


def test_chat_generation():
    """Test full chat generation pipeline."""
    print("\n" + "=" * 60)
    print("Testing Chat Generation")
    print("=" * 60)

    backend = create_backend(
        backend_type="orangepi",
        model_path="/path/to/model",
        gpu_id=0,
        backend_server_host="127.0.0.1",
        backend_server_port=50051,
    )

    print("\n[1/3] Connecting...")
    backend.load_model()
    print("✓ Connected")

    print("\n[2/3] Prefilling context...")
    msgs = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
    backend.chat_prefill(session_id="test-gen", msgs=msgs)
    print("✓ Prefill complete")

    print("\n[3/3] Generating response (streaming)...")
    try:
        for i, chunk in enumerate(backend.chat_streaming_generate(
            session_id="test-gen",
            generate_audio=False,
            max_new_tokens=50,
        )):
            print(f"  Chunk {i+1}: {chunk}")
            if i >= 4:  # Limit for testing
                break
        print("✓ Generation complete")
    except NotImplementedError as e:
        print(f"⚠ Streaming not yet implemented: {e}")
    except Exception as e:
        print(f"✗ Generation failed: {e}")

    backend.shutdown()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Orange Pi backend")
    parser.add_argument(
        "--test",
        choices=["connection", "generation", "all"],
        default="connection",
        help="Which test to run",
    )
    args = parser.parse_args()

    if args.test in ["connection", "all"]:
        success = test_backend_connection()
        if not success:
            sys.exit(1)

    if args.test in ["generation", "all"]:
        test_chat_generation()
