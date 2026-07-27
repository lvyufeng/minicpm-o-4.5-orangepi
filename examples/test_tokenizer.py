#!/usr/bin/env python3
"""Test tokenizer integration with backend."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend import create_backend


def test_tokenizer_flow():
    """Test full tokenize → prefill → generate → detokenize flow."""
    print("=" * 60)
    print("Testing Tokenizer Integration")
    print("=" * 60)

    # Model path (resolve to absolute path)
    model_path = str(Path(__file__).parent.parent / "models" / "MiniCPM-o-4.5")

    print("\n[1/5] Creating backend...")
    backend = create_backend(
        backend_type="orangepi",
        model_path=model_path,
        gpu_id=0,
        backend_server_host="127.0.0.1",
        backend_server_port=50051,
    )

    print("\n[2/5] Loading model and tokenizer...")
    try:
        backend.load_model()
        print("✓ Backend connected")

        if backend.tokenizer is None:
            print("✗ Tokenizer not loaded - make sure transformers is installed")
            print("  pip install transformers")
            return False

        print(f"✓ Tokenizer loaded: {type(backend.tokenizer).__name__}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

    print("\n[3/5] Testing tokenization...")
    test_text = "Hello, world!"
    try:
        tokens = backend.tokenizer.encode(test_text)
        decoded = backend.tokenizer.decode(tokens)
        print(f"  Input: '{test_text}'")
        print(f"  Tokens: {tokens} ({len(tokens)} tokens)")
        print(f"  Decoded: '{decoded}'")
        print("✓ Tokenization works")
    except Exception as e:
        print(f"✗ Tokenization failed: {e}")
        return False

    print("\n[4/5] Testing prefill with tokens...")
    try:
        msgs = [{"role": "user", "content": "Hi"}]
        prompt = backend.chat_prefill(session_id="test-tok", msgs=msgs)
        print(f"✓ Prefill successful: {prompt}")
    except Exception as e:
        print(f"✗ Prefill failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n[5/5] Testing generation...")
    try:
        for i, chunk in enumerate(backend.chat_streaming_generate(
            session_id="test-tok",
            generate_audio=False,
            max_new_tokens=20,
        )):
            print(f"  Chunk {i+1}: token_id={chunk['token_id']}, text='{chunk['text']}'")
            if i >= 5:  # Limit for testing
                break
        print("✓ Generation works")
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n[6/6] Shutting down...")
    backend.shutdown()

    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_tokenizer_flow()
    sys.exit(0 if success else 1)
